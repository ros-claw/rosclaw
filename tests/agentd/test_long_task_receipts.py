"""Long-session regressions; real journal/admission, fixture action transport only."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rosclaw.agentd.action_dispatch import HandlerOutcome
from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService
from rosclaw.agentd.pi_bridge.context import build_embodied_context
from rosclaw.contracts.agent.agent_event import AgentEventType
from tests.agentd.test_pi_tool_bridge import _request, _setup
from tests.agentd.test_seven3_policy_grant import _propose


@pytest.mark.parametrize(
    "variant",
    [
        "valid",
        "separate_receipt",
        "missing",
        "other_mission",
        "unverified",
        "failed",
        "wrong_receipt",
    ],
)
async def test_exact_receipt_after_fifty_events(tmp_path: Path, variant: str):
    service, mission = await _setup(tmp_path)
    try:
        for i in range(65):
            await service._events.append(mission.mission_id, AgentEventType.TOOL_PROGRESS, {"i": i})
        ctx, card = await _propose(
            service,
            mission,
            "ur5e.move_to_pose",
            idem="long-session",
            args={"x": 0.35, "y": 0.25, "z": 0.45},
        )
        assert (
            ActionAdmissionService(service).decision_status(card["approval_id"])["status"]
            == "APPROVED"
        )
        receipt_id = "receipt_exact" if variant == "separate_receipt" else "action_exact"

        async def dispatch(*args, **kwargs):
            payload = {
                "action_id": "action_exact",
                "final_state": "FAILED" if variant == "failed" else "COMPLETED",
                "verified": variant != "unverified",
            }
            if variant == "separate_receipt":
                payload["receipt_id"] = receipt_id
            if variant == "wrong_receipt":
                payload["receipt_id"] = "receipt_other"
            if variant != "missing":
                await service._events.append(
                    "mission_other" if variant == "other_mission" else mission.mission_id,
                    AgentEventType.RECEIPT_RECEIVED,
                    payload,
                )
            return HandlerOutcome(
                text="fixture transport terminal; no physical task score",
                terminal_receipt=True,
                evidence_ref="receipt://" + receipt_id,
            )

        with patch("rosclaw.agentd.action_dispatch.request_action", side_effect=dispatch):
            result = await ActionAdmissionService(service).execute(
                card["approval_id"], request=ctx, caller_pid=1, caller_uid=1000
            )
        assert result["executed"] is (variant in ["valid", "separate_receipt"]), result
        if result["executed"]:
            assert result["action_id"] == "action_exact"
    finally:
        await service.close()


async def test_context_contains_latest_receipts_after_busy_history(tmp_path: Path):
    service, mission = await _setup(tmp_path)
    try:
        for i in range(5):
            await service._events.append(
                mission.mission_id,
                AgentEventType.RECEIPT_RECEIVED,
                {"action_id": f"action_{i}", "verified": True, "final_state": "COMPLETED"},
            )
            for j in range(60):
                await service._events.append(
                    mission.mission_id, AgentEventType.TOOL_PROGRESS, {"j": j}
                )
        envelope = build_embodied_context(service, mission.mission_id)
        assert [r["action_id"] for r in envelope.receipts] == ["action_2", "action_3", "action_4"]
    finally:
        await service.close()


async def test_process_start_receipt_allows_independent_work(tmp_path: Path):
    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher

    service, mission = await _setup(tmp_path)
    try:
        result = await PiToolDispatcher(service).execute(
            _request(
                "rosclaw_process_start",
                mission=mission.mission_id,
                arguments={"command": "printf diagnostic-only"},
            )
        )
        import asyncio

        for _ in range(200):
            states = service._store.connection.execute("SELECT state FROM operations").fetchall()
            if states and all(r[0] in ["SUCCEEDED", "FAILED", "CANCELLED"] for r in states):
                break
            await asyncio.sleep(0.01)
        assert result.ok, result
        assert "现在请直接结束回合" not in result.summary
        assert "独立工作" in result.summary
    finally:
        await service.close()
