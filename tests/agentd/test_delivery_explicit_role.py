"""Delivery intent must be explicit before artifact reading/task admission."""
from __future__ import annotations

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


@pytest.mark.parametrize("arguments", [{}, {"role": ""}, {"role": "  "}, {"role": None}, {"role": 1}])
async def test_missing_delivery_intent_never_admits_or_reads(tmp_path, arguments, monkeypatch):
    service, mission = await _setup(tmp_path)
    try:
        dispatcher = PiToolDispatcher(service)
        async def forbidden_register(request):
            pytest.fail("invalid intent reached file-reading/admission entry")
        monkeypatch.setattr(dispatcher, "_artifact_register", forbidden_register)
        result = await dispatcher.execute(_request(
            "rosclaw_deliver", mission=mission.mission_id, idem="invalid_intent",
            lease=await _issue_lease(service, mission),
            arguments={"path": str(tmp_path / "nonexistent"), **arguments},
        ))
        assert not result.ok
        assert result.status == "REJECTED"
        assert result.error_code == "DELIVERY_ROLE_REQUIRED"
        assert service._task_kernel.latest_task_for(mission.mission_id, "pi_1") is None
    finally:
        await service.close()


async def test_omitted_role_does_not_finish_source_only_task(tmp_path):
    service, mission = await _setup(tmp_path)
    try:
        kernel = service._task_kernel
        bound = kernel.bind_message(
            mission_id=mission.mission_id, session_ref="pi_1", backend_native_id="pi_1",
            message_id="source_only", text="SOURCE_ONLY: do not finish; prepare source",
            cwd=str(tmp_path), body_id="",
        )
        path = tmp_path / "source.json"
        path.write_text('{"source_only": true}')
        result = await PiToolDispatcher(service).execute(_request(
            "rosclaw_deliver", mission=mission.mission_id, idem="missing_role_existing",
            lease=await _issue_lease(service, mission), arguments={"path": str(path)},
        ))
        assert not result.ok
        assert result.error_code == "DELIVERY_ROLE_REQUIRED"
        assert kernel.get_task(str(bound["task_id"]))["state"] == "RUNNING"
        assert kernel._conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    finally:
        await service.close()
