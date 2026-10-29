"""Software dispatcher fixture only; no Runtime, robot or native goal."""

from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from rosclaw.contracts.pi.tool_request import PiToolResultV1
from tests.agentd.test_pi_tool_bridge import _request, _setup


async def test_generic_unverified_failure_retains_the_deterministic_failure_guard(tmp_path: Path):
    service, mission = await _setup(tmp_path)
    dispatcher = PiToolDispatcher(service)
    calls = []

    async def blocked(request):
        calls.append(request.idempotency_key)
        return PiToolResultV1(
            request_id=request.request_id,
            ok=False,
            status="BLOCKED",
            summary="Fixture: no native dispatch",
            error_code="ACTION_NOT_VERIFIED",
        )

    dispatcher._execute_validated = blocked
    first = await dispatcher.execute(
        caller_pid=1,
        caller_uid=1000,
        request=_request(
            "rosclaw_request_action",
            mission=mission.mission_id,
            idem="first",
            arguments={"capability_id": "docking.dock", "arguments": {}},
        ),
    )
    second = await dispatcher.execute(
        caller_pid=1,
        caller_uid=1000,
        request=_request(
            "rosclaw_request_action",
            mission=mission.mission_id,
            idem="new-intent",
            arguments={"capability_id": "docking.dock", "arguments": {}},
        ),
    )
    assert first.error_code == "ACTION_NOT_VERIFIED"
    assert second.error_code == "DOOM_LOOP"
    assert calls == ["first"]
    await service.close()


async def test_known_idle_precondition_allows_new_intent_but_old_key_stays_cached(tmp_path: Path):
    service, mission = await _setup(tmp_path)
    dispatcher = PiToolDispatcher(service)
    calls = []
    changed = [False]

    async def execute(request):
        calls.append(request.idempotency_key)
        return PiToolResultV1(
            request_id=request.request_id,
            ok=changed[0],
            status="COMPLETED" if changed[0] else "BLOCKED",
            summary="Fixture only",
            error_code=None if changed[0] else "ACTION_PRECONDITION_NOT_MET",
        )

    dispatcher._execute_validated = execute

    def request(key):
        return _request(
            "rosclaw_request_action",
            mission=mission.mission_id,
            idem=key,
            arguments={"capability_id": "docking.dock", "arguments": {}},
        )

    first = await dispatcher.execute(caller_pid=1, caller_uid=1000, request=request("first"))
    assert first.status == "BLOCKED" and not first.ok and calls == ["first"]
    changed[0] = True  # Explicit software premise; production admission is unchanged.
    replay = await dispatcher.execute(caller_pid=1, caller_uid=1000, request=request("first"))
    assert replay.status == "BLOCKED" and not replay.ok and calls == ["first"]
    next_intent = await dispatcher.execute(
        caller_pid=1, caller_uid=1000, request=request("after-new-observation")
    )
    assert next_intent.ok is True and calls == ["first", "after-new-observation"]
    await service.close()


def canonical_block():
    return {
        "capability_id": "docking.dock",
        "action_id": "original",
        "final_state": "BLOCKED",
        "verified": False,
        "execution_mode": "SIMULATION",
        "evidence_domain": "SIMULATION",
        "dispatch_result": {"native_goal_dispatched": False},
        "verification_result": {
            "passed": False,
            "scope": "native_dock_predispatch_precondition_only",
            "original_native_terminal_known": True,
            "physical_stop_observed": True,
            "cleaner_disabled": True,
            "dock_visible": False,
        },
        "errors": [{"code": "NATIVE_DOCK_NOT_VISIBLE"}],
    }


def test_only_original_canonical_known_idle_negative_receipt_is_replannable():
    from rosclaw.agentd.action_dispatch import _known_idle_dock_precondition

    assert _known_idle_dock_precondition(canonical_block(), "docking.dock", "original")


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_state", "FAILED"),
        ("verified", True),
        ("execution_mode", "REAL"),
        ("evidence_domain", "REAL"),
        ("action_id", "other"),
        ("capability_id", "navigation.follow_path"),
        ("dispatch_result", None),
        ("verification_result", None),
        ("errors", []),
    ],
)
def test_unverified_failure_or_unknown_proof_does_not_gain_replan_class(field, value):
    from rosclaw.agentd.action_dispatch import _known_idle_dock_precondition

    receipt = canonical_block()
    receipt[field] = value
    assert not _known_idle_dock_precondition(receipt, "docking.dock", "original")


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", True),
        ("original_native_terminal_known", None),
        ("original_native_terminal_known", False),
        ("physical_stop_observed", False),
        ("cleaner_disabled", False),
        ("dock_visible", None),
        ("dock_visible", True),
        ("scope", "other"),
    ],
)
def test_all_original_idle_precondition_facts_are_required(field, value):
    from rosclaw.agentd.action_dispatch import _known_idle_dock_precondition

    receipt = canonical_block()
    receipt["verification_result"][field] = value
    assert not _known_idle_dock_precondition(receipt, "docking.dock", "original")


def test_actual_receipt_serializer_preserves_the_negative_evidence_contract():
    from rosclaw.agentd.action_dispatch import _known_idle_dock_precondition
    from rosclaw.kernel import ActionState, EvidenceLevel, ExecutionMode, ExecutionReceipt

    value = canonical_block()
    receipt = ExecutionReceipt(
        action_id="original",
        trace_id="software-fixture",
        mode=ExecutionMode.SIMULATION,
        body_id="fixture",
        body_snapshot_hash="b" * 64,
        capability_id="docking.dock",
        final_state=ActionState.BLOCKED,
        evidence_level=EvidenceLevel.REQUESTED,
        dispatch_result=value["dispatch_result"],
        verification_result=value["verification_result"],
        errors=value["errors"],
    )
    assert receipt.verified is False
    assert _known_idle_dock_precondition(receipt.to_dict(), "docking.dock", "original")
