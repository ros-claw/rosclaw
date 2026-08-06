"""MissionRunner — per-mission locks + continuous event loop (PR-03, 大纲 §4).

规则：

- 同一 Mission 顺序执行（每 Mission 一把锁），不同 Mission 可并发。
- 回合结束在 wait 状态（WAIT_APPROVAL 等）时注册 WakeConditionV1，
  外部事件（approval.decided / worker.completed / receipt.terminal /
  deadline）自动唤醒并续跑——用户不需要再说"继续"。
- 唤醒注入是明确标记的系统消息（``[rosclaw system wake]``），如实进入
  conversation journal——绝不伪装成用户发言。
- 连续唤醒链有上限（防止自激循环）；超时按 on_timeout 语义处理。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from rosclaw.contracts.agent.agent_event import AgentEventType
from rosclaw.contracts.agent.wake import WakeConditionV1

#: 回合结束后允许自动续跑的状态 → 对应唤醒类型。
_WAIT_STATE_WAKE = {
    "WAIT_APPROVAL": "approval_decided",
}
MAX_WAKE_CHAIN = 3


@dataclass
class _Pending:
    condition: WakeConditionV1
    event: asyncio.Event = field(default_factory=asyncio.Event)
    fired_payload: dict = field(default_factory=dict)


class MissionRunner:
    def __init__(self, service) -> None:
        self._service = service
        self._mission_locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, _Pending] = {}
        self._wake_tasks: dict[str, asyncio.Task] = {}

    def lock_for(self, mission_id: str) -> asyncio.Lock:
        if mission_id not in self._mission_locks:
            self._mission_locks[mission_id] = asyncio.Lock()
        return self._mission_locks[mission_id]

    # ------------------------------------------------------------------
    async def submit_turn(self, mission_id: str, text: str) -> str:
        """202-style submit with wake tracking (supersedes bare task spawn)."""
        service = self._service
        mission = service.store.get_mission(mission_id)
        if mission is None:
            from rosclaw.contracts.common import ValidationError

            raise ValidationError(f"unknown mission {mission_id!r}")
        existing = service._turn_tasks.get(mission_id)
        if existing is not None and not existing.done():
            from rosclaw.contracts.common import ValidationError

            raise ValidationError("a turn is already running for this mission")
        from rosclaw.contracts.common import new_id

        turn_id = new_id("turn")
        await service._events.append(
            mission_id, AgentEventType.TURN_ACCEPTED, {"text": text[:500]}, turn_id=turn_id
        )

        async def on_delta(piece: str) -> None:
            await service._events.append(
                mission_id, AgentEventType.MODEL_TEXT_DELTA, {"text": piece}, turn_id=turn_id
            )

        async def run() -> None:
            # send_turn 持有每 Mission 锁（不重入）；这里不再加锁。
            state_before = service.store.get_mission(mission_id).state
            await service._events.append(
                mission_id, AgentEventType.AGENT_STARTED, {}, turn_id=turn_id
            )
            try:
                result = await service.send_turn(mission_id, text, on_delta)
            except Exception as exc:  # noqa: BLE001 - errors become events
                await service._events.append(
                    mission_id, AgentEventType.AGENT_FAILED, {"error": str(exc)}, turn_id=turn_id
                )
                await service._events.append(
                    mission_id, AgentEventType.ERROR, {"error": str(exc)}, turn_id=turn_id
                )
                # settled 总能发出（TUI 停 spinner 的唯一可靠信号）。
                await service._events.append(
                    mission_id, AgentEventType.AGENT_SETTLED, {"outcome": "failed"}, turn_id=turn_id
                )
                return
            await service._events.append(
                mission_id,
                AgentEventType.TURN_ENDED,
                {"state": result.state.value},
                turn_id=turn_id,
            )
            await self._after_turn(mission_id, state_before, result, turn_id, chain=0)
            await service._events.append(
                mission_id,
                AgentEventType.AGENT_SETTLED,
                {"outcome": result.state.value},
                turn_id=turn_id,
            )

        service._turn_tasks[mission_id] = asyncio.create_task(run())
        return turn_id

    # ------------------------------------------------------------------
    async def _after_turn(self, mission_id, state_before, result, turn_id, *, chain: int) -> None:
        service = self._service
        if result.state is not state_before:
            await service._events.append(
                mission_id,
                AgentEventType.MISSION_STATE_CHANGED,
                {"from": state_before.value, "to": result.state.value},
                turn_id=turn_id,
            )
        if result.state.value == "IDLE":
            await service._events.append(
                mission_id, AgentEventType.MISSION_COMPLETED, {}, turn_id=turn_id
            )
            return
        if result.state.value == "FAILED":
            await service._events.append(
                mission_id, AgentEventType.MISSION_FAILED, {}, turn_id=turn_id
            )
            return
        wake_type = _WAIT_STATE_WAKE.get(result.state.value)
        if wake_type is None or chain >= MAX_WAKE_CHAIN:
            return  # user-yield state: wait for human input
        if result.state.value == "WAIT_APPROVAL":
            pending = service.pending_approvals(mission_id)
            reference_id = pending[0].request_id if pending else None
        else:
            reference_id = None
        condition = WakeConditionV1(
            type=wake_type,
            reference_id=reference_id,
            deadline=(datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
            on_timeout="WAIT_INPUT",
        )
        self._pending[mission_id] = _Pending(condition=condition)
        self._wake_tasks[mission_id] = asyncio.create_task(
            self._wait_and_resume(mission_id, turn_id, chain)
        )

    async def _wait_and_resume(self, mission_id: str, turn_id: str, chain: int) -> None:
        pending = self._pending.get(mission_id)
        if pending is None:
            return
        deadline = datetime.fromisoformat(pending.condition.deadline).timestamp()
        timeout = max(0.1, deadline - datetime.now(UTC).timestamp())
        try:
            await asyncio.wait_for(pending.event.wait(), timeout=timeout)
        except TimeoutError:
            self._pending.pop(mission_id, None)
            return  # approval 本身会过期（EXPIRED），无需伪造唤醒
        self._pending.pop(mission_id, None)
        notice = pending.fired_payload.get(
            "notice", "[rosclaw system wake] registered wake condition fired; proceed per protocol."
        )
        service = self._service
        mission = service.store.get_mission(mission_id)
        if mission is None or mission.state.value in ("IDLE", "FAILED"):
            return
        state_before = mission.state
        await service._events.append(
            mission_id, AgentEventType.AGENT_STARTED, {"wake": True}, turn_id=turn_id
        )
        try:
            result = await service.send_turn(mission_id, notice)
        except Exception:  # noqa: BLE001
            await service._events.append(
                mission_id, AgentEventType.AGENT_FAILED, {"error": "wake resume failed"},
                turn_id=turn_id,
            )
            await service._events.append(
                mission_id, AgentEventType.AGENT_SETTLED, {"outcome": "failed"}, turn_id=turn_id
            )
            return
        await self._after_turn(mission_id, state_before, result, turn_id, chain=chain + 1)
        await service._events.append(
            mission_id,
            AgentEventType.AGENT_SETTLED,
            {"outcome": result.state.value},
            turn_id=turn_id,
        )

    # ------------------------------------------------------------------
    # wake notifications (called by service/handlers on external events)
    # ------------------------------------------------------------------
    def notify_approval_decided(
        self, mission_id: str, request_id: str, *, approved: bool, grant_id: str | None
    ) -> None:
        pending = self._pending.get(mission_id)
        if (
            pending is None
            or pending.condition.type != "approval_decided"
            or pending.event.is_set()
        ):
            return
        if (
            pending.condition.reference_id is not None
            and pending.condition.reference_id != request_id
        ):
            return
        if approved:
            pending.fired_payload["notice"] = (
                f"[rosclaw system wake] approval {request_id} was APPROVED by the operator. "
                f"grant_id={grant_id}. Continue per protocol: if the plan requires the "
                "approved action, emit REQUEST_ACTION referencing this grant_id. Do not "
                "claim execution before a verified terminal receipt."
            )
        else:
            pending.fired_payload["notice"] = (
                f"[rosclaw system wake] approval {request_id} was DENIED by the operator. "
                "Do not retry the same action request; explain the blockage and propose "
                "a safer alternative or wait for user direction."
            )
        pending.event.set()

    def cancel_pending(self, mission_id: str) -> None:
        task = self._wake_tasks.pop(mission_id, None)
        if task is not None and not task.done():
            task.cancel()
        self._pending.pop(mission_id, None)
