"""Actual SIM channel with explicit synthetic MCP results; no robot or model."""

import asyncio
import json

import pytest

from rosclaw.agentd.sim_executor import SimActionChannel, SimActionError


class Transport:
    def __init__(self, value):
        self.value = value

    async def call_tool(self, name, arguments):
        return json.dumps(self.value)


def execute(value):
    channel = SimActionChannel(
        command="not-started", args=(), name="CPU-fixture", client=Transport(value)
    )
    return asyncio.run(
        channel.execute(
            capability_id="execute",
            arguments={},
            grant_id="synthetic-component-no-authority",
            mode="SIMULATION",
        )
    )


@pytest.mark.parametrize(
    "value",
    [
        {"query_succeeded": True, "succeeded": None, "status": "UNKNOWN"},
        {"query_succeeded": True, "succeeded": None, "status": "ACTIVE"},
        {"query_succeeded": True, "succeeded": True, "status": "RUNNING"},
        {"query_succeeded": True, "succeeded": False, "status": "SUCCEEDED"},
        {"query_succeeded": False, "status": "READY"},
    ],
)
def test_query_or_pending_motion_result_never_creates_terminal_success(value):
    with pytest.raises(SimActionError):
        execute(value)


@pytest.mark.parametrize(
    "value",
    [
        {"query_succeeded": True, "succeeded": True, "status": "SUCCEEDED"},
        {"ok": True, "position": [1, 2]},
        {
            "query_succeeded": True,
            "succeeded": True,
            "status": "SUCCEEDED",
            "driver_acknowledged": False,
        },
    ],
)
def test_known_success_and_existing_legacy_results_keep_driver_only_scope(value):
    result = execute(value)
    assert result.final_state == "COMPLETED"
    assert result.receipt["executor_result"] == value
    assert result.receipt["physical_effect_proven"] is False
    assert result.receipt["usable_for_real_execution"] is False
