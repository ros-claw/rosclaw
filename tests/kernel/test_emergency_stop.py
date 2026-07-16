"""Emergency-stop receipt and idempotency contracts."""

from __future__ import annotations

import time

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.kernel import ActionEnvelope, ActionState, EmergencyStopStatus, ExecutionMode


class _Driver:
    def __init__(self, result: object = None) -> None:
        self.result = result
        self.calls = 0

    def emergency_stop(self) -> object:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _runtime() -> Runtime:
    return Runtime(
        RuntimeConfig(
            enable_event_persistence=False,
            enable_tracing=False,
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
        )
    )


def test_stop_dispatch_is_not_reported_as_physical_stop() -> None:
    runtime = _runtime()
    driver = _Driver(None)
    runtime.register_driver("arm", driver)

    receipt = runtime.request_emergency_stop("operator request", request_id="stop-1")

    assert receipt.request_dispatched is True
    assert receipt.driver_acknowledged is False
    assert receipt.physical_stop_observed is False
    assert receipt.stopped is False
    assert receipt.final_status is EmergencyStopStatus.UNVERIFIED


def test_stop_requires_every_target_to_be_physically_verified() -> None:
    runtime = _runtime()
    verified = _Driver(
        {
            "acknowledged": True,
            "physical_stop_observed": True,
            "observed_joint_velocity": [0.0] * 6,
            "verification_source": "simulator_state",
        }
    )
    unverified = _Driver({"acknowledged": True})
    runtime.register_driver("arm", verified)
    runtime.register_driver("gripper", unverified)

    receipt = runtime.request_emergency_stop("collision", request_id="stop-2")

    assert receipt.driver_acknowledged is True
    assert receipt.physical_stop_observed is False
    assert receipt.stopped is False
    assert receipt.final_status is EmergencyStopStatus.ACKNOWLEDGED
    assert receipt.acknowledged_drivers == ["arm", "gripper"]


def test_physically_verified_stop_is_truthfully_reported() -> None:
    runtime = _runtime()
    driver = _Driver(
        {
            "acknowledged": True,
            "physical_stop_observed": True,
            "observed_velocity": 0.0,
            "observed_joint_velocity": [0.0] * 6,
            "verification_source": "joint_state_feedback",
        }
    )
    runtime.register_driver("arm", driver)

    receipt = runtime.request_emergency_stop("collision", request_id="stop-3")

    assert receipt.stopped is True
    assert receipt.final_status is EmergencyStopStatus.PHYSICALLY_VERIFIED
    assert receipt.verification_source == "joint_state_feedback"


def test_repeated_stop_request_is_idempotent() -> None:
    runtime = _runtime()
    driver = _Driver({"acknowledged": True})
    runtime.register_driver("arm", driver)

    first = runtime.request_emergency_stop("first", request_id="stop-repeat")
    second = runtime.request_emergency_stop("duplicate", request_id="stop-repeat")

    assert first.to_dict() == second.to_dict()
    assert driver.calls == 1


def test_stop_without_runtime_drivers_fails_closed() -> None:
    receipt = _runtime().request_emergency_stop("no drivers", request_id="stop-none")

    assert receipt.request_dispatched is False
    assert receipt.stopped is False
    assert receipt.final_status is EmergencyStopStatus.FAILED


def test_stop_timeout_is_reported_without_acknowledgement() -> None:
    class SlowDriver:
        def emergency_stop(self) -> dict[str, object]:
            time.sleep(0.05)
            return {"acknowledged": True, "physical_stop_observed": False}

    runtime = _runtime()
    runtime.register_driver("slow", SlowDriver())

    receipt = runtime.request_emergency_stop(
        "timeout injection",
        request_id="stop-timeout",
        timeout_sec=0.005,
    )

    assert receipt.timeout is True
    assert receipt.driver_acknowledged is False
    assert receipt.final_status is EmergencyStopStatus.UNVERIFIED
    assert receipt.errors[0]["code"] == "DRIVER_ACK_TIMEOUT"


def test_stop_partial_ack_is_distinct_from_full_ack() -> None:
    runtime = _runtime()
    runtime.register_driver("arm", _Driver({"acknowledged": True}))
    runtime.register_driver("gripper", _Driver({"acknowledged": False}))

    receipt = runtime.request_emergency_stop("partial", request_id="stop-partial")

    assert receipt.driver_acknowledged is False
    assert receipt.acknowledged_drivers == ["arm"]
    assert receipt.unacknowledged_drivers == ["gripper"]
    assert receipt.final_status is EmergencyStopStatus.PARTIALLY_ACKNOWLEDGED


def test_stop_latch_blocks_subsequent_real_action() -> None:
    runtime = _runtime()
    runtime.request_emergency_stop("operator", request_id="stop-latch")
    action = ActionEnvelope(
        actor_id="pytest",
        agent_framework="pytest",
        session_id="stop-latch",
        body_id="real-arm",
        capability_id="arm.move",
        arguments={},
        execution_mode=ExecutionMode.REAL,
    )

    receipt = runtime.submit_action(action)

    assert receipt.final_state is ActionState.BLOCKED
    assert receipt.errors[0]["code"] == "EMERGENCY_STOP_LATCHED"
