"""Authority snapshots, real lease DB and admission; action transport is fixture-only."""

from unittest.mock import AsyncMock, patch

import pytest

from rosclaw.agentd.pi_bridge.action_admission import ActionAdmissionService, ActionRequestContext
from rosclaw.agentd.pi_bridge.context import build_embodied_context
from rosclaw.agentd.pi_bridge.context_lease import (
    ContextLeaseStore,
    action_dependency_hashes,
    context_hash_of,
)
from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError
from tests.agentd.test_pi_tool_bridge import _setup

CAP = "ur5e.move_to_pose"


def facts():
    return {
        "status": "OBSERVED",
        "body_hash": "observed-body",
        "global_hazards": [],
        "capability_facts": {
            CAP: {"readiness": "AVAILABLE", "blocking_issues": []},
            "unrelated.debug": {"readiness": "UNKNOWN"},
        },
        "blocking_issues": [],
    }


def issue(service, mission):
    envelope = build_embodied_context(service, mission.mission_id)
    bindings = SessionBindingStore(service._store.connection)
    binding = bindings.binding_for_session("pi_1")
    writer = bindings.writer_of(mission.mission_id)
    lease = ContextLeaseStore(service._store.connection).issue(
        pi_session_id="pi_1",
        mission_id=mission.mission_id,
        context_revision=envelope.context_revision,
        context_hash=context_hash_of(envelope),
        body_hash=mission.body_binding.effective_body_hash,
        mode=mission.mode.value,
        binding_id=binding.binding_id,
        writer_lease_id=writer.lease_id,
        caller_pid=1,
        caller_uid=1000,
        dependency_hashes=action_dependency_hashes(envelope),
    )
    return ActionRequestContext(
        "pi_1",
        mission.mission_id,
        envelope.context_revision,
        mission.body_binding.effective_body_hash,
        mission.mode.value,
        "dependency-test",
        lease.context_lease_id,
    )


@pytest.mark.parametrize("change", ["unrelated", "prose", "required", "global", "body_identity"])
async def test_scoped_context_changes_have_real_lease_semantics(tmp_path, monkeypatch, change):
    service, mission = await _setup(tmp_path)
    current = facts()

    def observed(*_):
        return {
            "status": "OBSERVED",
            "summary": current.get("prose", "original"),
            "admission_facts": current,
        }

    monkeypatch.setattr("rosclaw.connectors.ros.context.native.native_ros_observations", observed)
    try:
        request = issue(service, mission)
        admission = ActionAdmissionService(service)
        admission._validate_request_context(
            request, caller_pid=1, caller_uid=1000, capability_id=CAP
        )
        if change == "unrelated":
            current["capability_facts"]["unrelated.debug"]["readiness"] = "BLOCKED"
        if change == "prose":
            current["prose"] = "explanation changed"
        if change == "required":
            current["capability_facts"][CAP]["readiness"] = "UNKNOWN"
        if change == "global":
            current["global_hazards"] = [{"code": "ROS_TF_006", "identity": {}}]
        if change == "body_identity":
            current["body_hash"] = "different-body"
        if change in ["required", "global", "body_identity"]:
            with pytest.raises(ToolBridgeError):
                admission._validate_request_context(
                    request, caller_pid=1, caller_uid=1000, capability_id=CAP
                )
        else:
            admission._validate_request_context(
                request, caller_pid=1, caller_uid=1000, capability_id=CAP
            )
    finally:
        await service.close()


async def test_dependency_change_after_approval_never_dispatches_or_consumes_grant(
    tmp_path, monkeypatch
):
    service, mission = await _setup(tmp_path)
    current = facts()
    monkeypatch.setattr(
        "rosclaw.connectors.ros.context.native.native_ros_observations",
        lambda *_: {"status": "OBSERVED", "admission_facts": current},
    )
    try:
        request = issue(service, mission)
        admission = ActionAdmissionService(service)
        card = await admission.propose(
            request=request,
            capability_id=CAP,
            arguments={"x": 0.35, "y": 0.25, "z": 0.45},
            expected_effect="fixture transport",
            risk_tier="LOW",
            caller_pid=1,
            caller_uid=1000,
        )
        assert admission.decision_status(card["approval_id"])["status"] == "APPROVED"
        current["capability_facts"][CAP]["readiness"] = "BLOCKED"
        dispatch = AsyncMock()
        with (
            patch("rosclaw.agentd.action_dispatch.request_action", dispatch),
            pytest.raises(ToolBridgeError, match="dependency facts changed"),
        ):
            await admission.execute(
                card["approval_id"], request=request, caller_pid=1, caller_uid=1000
            )
        dispatch.assert_not_called()
        row = service._store.connection.execute(
            "SELECT consumed FROM mission_grants WHERE request_id=?", (card["approval_id"],)
        ).fetchone()
        assert not row["consumed"]
    finally:
        await service.close()
