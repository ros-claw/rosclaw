"""Transport loss after dispatch must not replay an unknown tool effect."""
import asyncio
from types import SimpleNamespace

import pytest

from rosclaw.agentd.tooling.persistent_client import PersistentMcpClient
from rosclaw.contracts.common import ValidationError


class FakeClient(PersistentMcpClient):
    def __init__(self, outcomes, initialization=None, reset_error=None):
        super().__init__(command="never-spawn", args=())
        self.outcomes = list(outcomes)
        self.initialization = list(initialization or [])
        self.ensure_count = self.dispatch_count = self.reset_count = 0
        self.reset_error = reset_error

    async def _ensure(self):
        self.ensure_count += 1
        if self.initialization:
            error = self.initialization.pop(0)
            if error:
                raise error
        return self

    async def call_tool_effect(self, name, arguments):
        self.dispatch_count += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def _reset(self, key):
        self.reset_count += 1
        if self.reset_error:
            raise self.reset_error


def result(error=False):
    return SimpleNamespace(isError=error, content=[SimpleNamespace(text="effect-result")])


def invoke(client):
    # Actual wrapper, fake SDK session: dispatches are observable side effects.
    class Session:
        call_tool = client.call_tool_effect

    original = client._ensure

    async def ensure():
        await original()
        return Session()

    client._ensure = ensure
    return asyncio.run(PersistentMcpClient.call_tool(client, "move", {}))


@pytest.mark.parametrize("reset_error", [None, RuntimeError("reset failed")])
def test_after_effect_disconnect_is_unknown_not_replayed(reset_error):
    loss = ConnectionResetError("response lost after effect")
    client = FakeClient([loss, result()], reset_error=reset_error)
    with pytest.raises(ValidationError, match="OUTCOME_UNCONFIRMED") as caught:
        invoke(client)
    assert caught.value.code == "MCP_TOOL_OUTCOME_UNCONFIRMED"
    assert caught.value.__cause__ is loss
    assert client.dispatch_count == client.reset_count == 1


def test_initialization_reconnect_dispatches_once():
    client = FakeClient([result()], [ConnectionError("before dispatch"), None])
    assert invoke(client) == "effect-result"
    assert (client.ensure_count, client.dispatch_count, client.reset_count) == (2, 1, 1)


def test_initialization_retry_is_bounded_without_dispatch():
    client = FakeClient([], [ConnectionError(), ConnectionError()])
    with pytest.raises(ValidationError, match="before tool dispatch"):
        invoke(client)
    assert client.ensure_count == 2 and client.dispatch_count == 0


@pytest.mark.parametrize("error", [ValidationError("invalid arguments"), asyncio.CancelledError()])
def test_dispatch_argument_error_or_cancellation_preserved(error):
    client = FakeClient([error])
    with pytest.raises(type(error)) as caught:
        invoke(client)
    assert caught.value is error
    assert client.dispatch_count == 1 and client.reset_count == 0


def test_structured_tool_error_not_replayed():
    client = FakeClient([result(error=True)])
    with pytest.raises(ValidationError, match="mcp tool move error: effect-result"):
        invoke(client)
    assert client.dispatch_count == 1 and client.reset_count == 0


@pytest.mark.parametrize("error", [ValidationError("bad setup"), asyncio.CancelledError()])
def test_initialization_validation_or_cancellation_preserved(error):
    client = FakeClient([], [error])
    with pytest.raises(type(error)) as caught:
        invoke(client)
    assert caught.value is error
    assert client.ensure_count == 1 and client.dispatch_count == client.reset_count == 0


def test_same_loop_calls_remain_serial_after_unknown_outcome():
    async def run():
        client = PersistentMcpClient(command="never-spawn", args=())
        active = peak = calls = resets = 0

        class Session:
            async def call_tool(self, name, arguments):
                nonlocal active, peak, calls
                calls += 1
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                if calls == 1:
                    raise ConnectionResetError("first response lost")
                return result()

        async def ensure():
            return Session()

        async def reset(key):
            nonlocal resets
            resets += 1

        client._ensure = ensure
        client._reset = reset
        results = await asyncio.gather(
            client.call_tool("first", {}), client.call_tool("second", {}),
            return_exceptions=True,
        )
        assert isinstance(results[0], ValidationError)
        assert results[0].code == "MCP_TOOL_OUTCOME_UNCONFIRMED"
        assert results[1] == "effect-result"
        assert (peak, calls, resets) == (1, 2, 1)

    asyncio.run(run())
