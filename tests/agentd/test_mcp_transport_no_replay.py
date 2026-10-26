"""A transport exception after a server side effect must not retry the physical call."""

from types import SimpleNamespace

import pytest

from rosclaw.agentd.tooling.persistent_client import PersistentMcpClient
from rosclaw.contracts.common import ValidationError


async def test_lost_result_is_not_replayed_and_next_call_reconnects():
    client = PersistentMcpClient(command="unused", args=())
    dispatched = []
    resets = []
    connections = []

    async def call(name, arguments):
        dispatched.append((name, arguments))
        if len(dispatched) == 1:
            raise ConnectionResetError("Server effect happened; response lost")
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text='{"observed":true}')])

    async def ensure():
        connections.append(True)
        return SimpleNamespace(call_tool=call)

    async def reset(key):
        resets.append(key)

    client._ensure = ensure
    client._reset = reset
    with pytest.raises(ValidationError, match="not.*replay|unknown"):
        await client.call_tool("physical_action", {"value": 1})
    assert len(dispatched) == 1 and len(resets) == 1
    # A new explicit observation is allowed to create a new connection; it is
    # never an automatic resubmission of the failed physical intention.
    result = await client.call_tool("read_status", {"exact_action_id": "original"})
    assert result == '{"observed":true}'
    assert [name for name, _ in dispatched] == ["physical_action", "read_status"]
