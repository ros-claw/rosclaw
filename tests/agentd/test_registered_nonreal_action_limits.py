"""Trusted SIM deadlines do not replace permission or original parent identity."""

import threading

import pytest

from rosclaw.agentd import action_dispatch
from rosclaw.agentd.action_channel import ActionChannelError

from .test_nonreal_action_session_lifecycle import Client, channel

CAP = "coverage.program_finite_v1"
LIMITS = {
    "response_wait_sec": 310.0,
    "parent_timeout_sec": 330.0,
    "verification_timeout_sec": 310.0,
}


def options(raw, mode="SIMULATION", capability=CAP, parent=330.0):
    # Existing dispatcher has only a legacy coverage.execute_slice override.
    function = getattr(action_dispatch, "_nonreal_response_options", lambda *a, **kw: {})
    return function(raw, mode=mode, capability=capability, timeout_sec=parent)


def test_registered_capability_uses_original_three_separate_deadlines():
    assert options({"nonreal_action_limits": {CAP: LIMITS}}) == {
        "response_wait_sec": 310.0,
        "verification_timeout_sec": 310.0,
    }
    assert options({}) == {}
    assert options({"nonreal_action_limits": {CAP: LIMITS}}, capability="motion.twist") == {}
    assert options({"coverage_response_wait_sec": 310}, capability="coverage.execute_slice") == {
        "response_wait_sec": 310.0
    }


def test_generic_registration_cannot_redefine_the_legacy_slice_contract():
    with pytest.raises(ActionChannelError, match="legacy"):
        options(
            {"nonreal_action_limits": {"coverage.execute_slice": LIMITS}},
            capability="coverage.execute_slice",
        )


@pytest.mark.parametrize(
    "mode,parent,change",
    [
        ("REAL", 330.0, {}),
        ("SHADOW", 330.0, {}),
        ("SIMULATION", 300.0, {}),
        ("SIMULATION", 330.0, {"parent_timeout_sec": 331.0}),
        ("SIMULATION", 330.0, {"verification_timeout_sec": 331.0}),
        ("SIMULATION", 330.0, {"response_wait_sec": 331.0}),
        ("SIMULATION", 330.0, {"response_wait_sec": True}),
        ("SIMULATION", 330.0, {"verification_timeout_sec": float("nan")}),
        ("SIMULATION", 330.0, {"verification_timeout_sec": 309.0}),
    ],
)
def test_invalid_or_non_sim_limits_refuse_before_any_remote_effect(mode, parent, change):
    with pytest.raises(ActionChannelError):
        options({"nonreal_action_limits": {CAP: dict(LIMITS, **change)}}, mode=mode, parent=parent)


async def test_early_common_response_retains_one_parent_and_310_verification():
    client = Client()
    release = threading.Event()

    def wait(*a, **kw):
        assert release.wait(2)
        return {"state": "FINISHED"}

    client.wait_for_action = wait
    service = channel(client)
    try:
        with pytest.raises(ActionChannelError, match="ACTION_RESULT_UNKNOWN"):
            await service.request_nonreal_action(
                capability_id=CAP,
                arguments={},
                grant_id="original_grant",
                action_id="original_parent",
                timeout_sec=330.0,
                response_wait_sec=0.03,
                verification_timeout_sec=310.0,
            )
        original = client.actions[0]
        assert (
            original.action_id == "original_parent"
            and original.verification_policy.timeout_sec == 310.0
        )
        deadline = original.deadline_at
        assert not client.closed
        with pytest.raises(ActionChannelError, match="still owned"):
            await service.request_nonreal_action(
                capability_id=CAP, arguments={}, grant_id="another_grant"
            )
        assert len(client.actions) == 1
        release.set()
        assert await service._owned_nonreal_task == "verified"
        assert original.deadline_at == deadline and client.closed == [original.session_id]
    finally:
        release.set()
        await service.close()
