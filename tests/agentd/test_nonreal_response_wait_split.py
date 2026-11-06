"""Response timeout is not loss of the original canonical action owner."""

import threading

import pytest

from rosclaw.agentd.action_channel import ActionChannelError

from .test_nonreal_action_session_lifecycle import Client, channel


async def test_early_response_retains_original_session_goal_and_parent_deadline():
    client = Client()
    entered = threading.Event()
    release = threading.Event()

    def wait(action_id, **kw):
        entered.set()
        assert release.wait(2)
        return {"state": "FINISHED"}

    client.wait_for_action = wait
    service = channel(client)
    try:
        with pytest.raises(ActionChannelError, match="ACTION_RESULT_UNKNOWN") as caught:
            await service.request_nonreal_action(
                capability_id="coverage.execute_slice",
                arguments={},
                grant_id="original-grant",
                timeout_sec=330,
                response_wait_sec=0.05,
                action_id="original-parent",
            )
        assert entered.is_set() and "grant_id=original-grant" in str(caught.value)
        original = client.actions[0]
        deadline = original.deadline_at
        assert original.action_id == "original-parent" and not client.closed
        assert original.verification_policy.timeout_sec == 330
        with pytest.raises(ActionChannelError, match="still owned"):
            await service.request_nonreal_action(
                capability_id="motion.twist", arguments={}, grant_id="another-grant"
            )
        assert len(client.actions) == 1
        release.set()
        assert await service._owned_nonreal_task == "verified"
        assert original.deadline_at == deadline and client.closed == [original.session_id]
    finally:
        release.set()
        await service.close()


async def test_service_close_revokes_retained_session_without_resubmission():
    client = Client()
    release = threading.Event()

    def wait(*_, **kwargs):
        release.wait(1)
        return {"state": "FINISHED"}

    client.wait_for_action = wait
    service = channel(client)
    try:
        with pytest.raises(ActionChannelError, match="ACTION_RESULT_UNKNOWN"):
            await service.request_nonreal_action(
                capability_id="coverage.execute_slice",
                arguments={},
                grant_id="grant",
                timeout_sec=330,
                response_wait_sec=0.02,
            )
        await service.close()
        assert client.closed == [client.actions[0].session_id] and len(client.actions) == 1
    finally:
        release.set()
