"""A delivery-first task uses the workspace supplied by the native tool relay."""

from pathlib import Path

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


@pytest.mark.parametrize("relative", [False, True])
async def test_first_delivery_binds_native_workspace(tmp_path, relative):
    service, mission = await _setup(tmp_path)
    try:
        workspace = tmp_path / "native 工作区"
        workspace.mkdir()
        source = workspace / "source.py"
        source.write_text("# native source\n")
        dispatcher = PiToolDispatcher(service)
        lease = await _issue_lease(service, mission)
        result = await dispatcher.execute(
            _request(
                "rosclaw_deliver",
                mission=mission.mission_id,
                lease=lease,
                arguments={
                    "path": source.name if relative else str(source),
                    "cwd": str(workspace),
                    "role": "diagnostic_source",
                },
            )
        )
        assert result.ok, result.summary
        kernel = service._task_kernel
        task = kernel.latest_task_for(mission.mission_id, "pi_1")
        assert Path(task["workspace_path"]).resolve() == workspace.resolve()
        assert task["state"] == "RUNNING"
        artifact = kernel._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (result.artifact_refs[0],),
        ).fetchone()
        assert artifact["task_id"] == task["task_id"]
        assert Path(artifact["path"]).resolve() == source.resolve()
    finally:
        await service.close()


async def test_existing_task_workspace_is_not_rebound_by_later_delivery(tmp_path):
    service, mission = await _setup(tmp_path)
    try:
        workspace = tmp_path / "bound_workspace"
        workspace.mkdir()
        kernel = service._task_kernel
        bound = kernel.ensure_task_for_effect(
            mission_id=mission.mission_id,
            session_ref="pi_1",
            backend_native_id="pi_1",
            cwd=str(workspace),
        )
        other = tmp_path / "supporting_sources"
        other.mkdir()
        source = other / "support.py"
        source.write_text("# supporting source\n")
        result = await PiToolDispatcher(service).execute(
            _request(
                "rosclaw_deliver",
                mission=mission.mission_id,
                lease=await _issue_lease(service, mission),
                arguments={"path": source.name, "cwd": str(other), "role": "diagnostic"},
            )
        )
        assert result.ok, result.summary
        task = kernel.get_task(bound["task_id"])
        assert Path(task["workspace_path"]).resolve() == workspace.resolve()
        assert task["state"] == "RUNNING"
        assert task["active_revision"] == bound["revision"]
    finally:
        await service.close()
