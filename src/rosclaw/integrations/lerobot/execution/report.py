"""Execution run report (plan §7.1 ``execution/report.py``)."""

from __future__ import annotations

import json
from typing import Any

from rosclaw.integrations.lerobot.execution.schema import ActionExecutionResult
from rosclaw.integrations.lerobot.execution.state import ExecutionState


class ExecutionReport:
    """Accumulate per-step results and render the execution summary."""

    def __init__(self, *, body_id: str, task: str):
        self.body_id = body_id
        self.task = task
        self.results: list[ActionExecutionResult] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.final_state: str = ExecutionState.DISARMED.value

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    def record_result(self, result: ActionExecutionResult) -> None:
        self.results.append(result)

    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        completed = sum(1 for r in self.results if r.status == "completed")
        blocked = sum(1 for r in self.results if r.status == "blocked")
        faults = sum(1 for r in self.results if r.status in {"fault", "aborted"})
        stale = sum(1 for r in self.results if r.status == "stale_action")
        sent = sum(1 for r in self.results if r.command_sent)
        acked = sum(1 for r in self.results if r.command_acknowledged)
        over_contact = sum(
            1
            for r in self.results
            if any("force_hard_limit" in d for d in r.verification.details)
        )
        protection = sum(
            1
            for r in self.results
            if any("status_protection" in d for d in r.verification.details)
        )
        return {
            "body_id": self.body_id,
            "task": self.task,
            "steps": len(self.results),
            "completed": completed,
            "blocked": blocked,
            "faults": faults,
            "stale_actions": stale,
            "commands_sent": sent,
            "commands_acknowledged": acked,
            "over_contact": over_contact,
            "hardware_protection": protection,
            "final_state": self.final_state,
        }

    def render_markdown(self) -> str:
        s = self.summary()
        lines = [
            "# P5 RH56 Execution Report",
            "",
            f"- Body: `{s['body_id']}`",
            f"- Task: `{s['task']}`",
            f"- Final state: `{s['final_state']}`",
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(s, indent=2),
            "```",
            "",
            "## Steps",
            "",
            "| # | status | sent | ack | max pos err | max force | max temp |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(self.results):
            max_err = max((e for e in r.position_error if e == e), default=0.0)
            max_force = max(r.force, default=0.0)
            max_temp = max(r.temperature, default=0.0)
            lines.append(
                f"| {i} | {r.status} | {r.command_sent} | {r.command_acknowledged} "
                f"| {max_err:.1f} | {max_force:.1f} | {max_temp:.1f} |"
            )
        return "\n".join(lines)
