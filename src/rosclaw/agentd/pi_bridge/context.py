"""EmbodiedContextEnvelope 构建（PR-PNA-2）：每轮从权威源现取现算。

绝不从缓存/session 历史取具身事实——Body/Self/Mission/approval/
action/safety 每轮重新生成，附 TTL 与内容 hash。
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from rosclaw.contracts.pi.embodied_context import EmbodiedContextEnvelopeV1

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService

ENVELOPE_TTL_SEC = 30.0


def build_embodied_context(service: AgentService, mission_id: str) -> EmbodiedContextEnvelopeV1:
    """从 MissionStore/BodySource/broker 现取最新具身事实。"""
    mission = service.get_mission(mission_id)
    if mission is None:
        raise ValueError(f"unknown mission {mission_id!r}")
    now = datetime.now(UTC)
    body = service._body_source.get_body(service._body_id)
    snapshot = service.snapshot(mission_id)
    pending = service.pending_approvals(mission_id)
    envelope = EmbodiedContextEnvelopeV1(
        mission_id=mission_id,
        context_revision=snapshot.context_revision,
        generated_at=now.isoformat(),
        expires_at=(now + timedelta(seconds=ENVELOPE_TTL_SEC)).isoformat(),
        body={
            "body_id": mission.body_binding.body_id,
            "effective_body_hash": mission.body_binding.effective_body_hash,
            "summary": body.summary if body else "body unavailable (fail closed)",
            "calibrated": body.calibrated if body else False,
            "issues": list(body.issues) if body else ["body source unavailable"],
        },
        self_state={
            "authorization_profile": service.authorization_profile(),
            "turn_in_flight": snapshot.turn_in_flight,
        },
        task_graph={
            "goal": mission.goal.text if mission.goal else "",
            "state": mission.state.value,
        },
        capabilities=sorted(
            d.tool_id
            for d in service._tool_catalog._descriptors.values()
            if d.execution_class.value == "OBSERVE"
            and service._tool_catalog.quarantine_reason(d.tool_id) is None
        )[:50],
        active_actions=[],
        receipts=[
            e.payload
            for e in service.events_replay(mission_id, limit=50)
            if e.type.value == "receipt.received"
        ][-3:],
        workers=[
            {"work_order_id": o.work_order_id, "assigned_to": o.assigned_to, "status": o.status}
            for o in service._worker_manager.orders_for_mission(mission_id)
        ][:10],
        pending_approvals=[
            {
                "request_id": r.request_id,
                "title": r.action_display.title,
                "risk_tier": r.action_display.risk_tier,
                "expires_at": r.expires_at,
            }
            for r in pending
        ],
        memory_summary={"note": "approved evidence pipeline; empty in SIM profile"},
        safety={
            "mode": mission.mode.value,
            "estop_note": "E-Stop 是独立 operator 路径，不经过本 Agent",
        },
        tool_policy={
            "allowed_tools": [
                "rosclaw_status",
                "rosclaw_observe",
                "rosclaw_verify",
                "rosclaw_memory_query",
                "rosclaw_fail_safe",
                "rosclaw_delegate",
                "rosclaw_request_action",
            ]
        },
        freshness={"generated_at": now.isoformat(), "ttl_sec": ENVELOPE_TTL_SEC},
    )
    envelope.hash = envelope_hash(envelope)
    return envelope


def envelope_hash(envelope: EmbodiedContextEnvelopeV1) -> str:
    """NA-FIX-1：RFC 8785 canonical JSON——Python/TS 逐字节一致。"""
    from rosclaw.contracts.pi.canonical import canonical_dumps

    payload = envelope.model_dump(mode="json")
    payload.pop("hash", None)
    canonical = canonical_dumps(payload)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:32]
