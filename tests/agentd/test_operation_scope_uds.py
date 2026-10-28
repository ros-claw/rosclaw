"""Actual authenticated Native bridge transport with finite host-only children."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.server import PiBridgeServer
from tests.agentd.test_operation_task_scope import start_owned_process
from tests.agentd.test_pi_tool_bridge import _request, _setup


@pytest.mark.parametrize("case", ["foreign_operation", "cached_result"])
async def test_authenticated_unix_tool_boundary_preserves_scope(tmp_path, case):
    service, mission = await _setup(tmp_path)
    with tempfile.TemporaryDirectory(prefix="rcscope-") as folder:
        bridge = PiBridgeServer(service, Path(folder) / "bridge.sock")
        writer = None
        try:
            await bridge.start()
            reader, writer = await asyncio.open_unix_connection(str(Path(folder) / "bridge.sock"))

            async def call(request, *, authorized=True):
                writer.write(
                    json.dumps(
                        {
                            "method": "pi.tools.execute",
                            "params": {
                                "token": service.control_token if authorized else "invalid-token",
                                "request": request.model_dump(),
                            },
                        }
                    ).encode()
                    + b"\n"
                )
                await writer.drain()
                return json.loads(await asyncio.wait_for(reader.readline(), 3))

            if case == "foreign_operation":
                operation = await start_owned_process(service, mission, "pi_foreign")
                request = _request(
                    "rosclaw_process_stop",
                    mission=mission.mission_id,
                    arguments={"operation_id": operation["operation_id"]},
                )
                denied = await call(request, authorized=False)
                assert denied["code"] == "UNAUTHORIZED"
                result = await call(request)
                assert not result["ok"] and result["code"] == "NOT_FOUND"
                assert (
                    service._operation_manager.get(operation["operation_id"])["state"] == "RUNNING"
                )
            else:
                request = _request("rosclaw_status", mission=mission.mission_id, idem="wire_key")
                original = await call(request)
                assert original["ok"]
                result = await call(request.model_copy(update={"mission_id": "foreign"}))
                assert not result["ok"] and result["code"] == "IDEMPOTENCY_CONFLICT"
                assert original["result"]["summary"] not in result["result"]["summary"]
                assert await call(request) == original
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            await bridge.stop()
            await service.close()
