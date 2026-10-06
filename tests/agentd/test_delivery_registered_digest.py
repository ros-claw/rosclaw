"""Delivery exposes the kernel's registered digest without a second tool call."""

import hashlib

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _issue_lease, _request, _setup


async def test_progress_delivery_returns_registered_digest_and_idempotent_ref(tmp_path):
    service, mission = await _setup(tmp_path)
    try:
        content = "原生源码证据\n".encode()
        path = tmp_path / "source.py"
        path.write_bytes(content)
        dispatcher = PiToolDispatcher(service)
        lease = await _issue_lease(service, mission)
        results = []
        for idem in ["source_first", "source_again"]:
            result = await dispatcher.execute(
                _request(
                    "rosclaw_deliver",
                    mission=mission.mission_id,
                    idem=idem,
                    lease=lease,
                    arguments={"path": str(path), "role": "diagnostic_source"},
                )
            )
            assert result.ok, result.summary
            results.append(result)
        digest = hashlib.sha256(content).hexdigest()
        assert results[0].artifact_refs == results[1].artifact_refs
        assert len(results[0].artifact_refs) == 1
        kernel = service._task_kernel
        rows = kernel._conn.execute("SELECT * FROM artifacts").fetchall()
        assert len(rows) == 1
        assert rows[0]["sha256"] == digest
        assert rows[0]["size_bytes"] == len(content)
        for result in results:
            assert f"sha256={digest}" in result.summary
            assert result.artifact_refs == [rows[0]["artifact_id"]]
        task = kernel.latest_task_for(mission.mission_id, "pi_1")
        assert task["state"] == "RUNNING"
    finally:
        await service.close()


async def test_post_terminal_append_returns_digest_without_reviving_task(tmp_path):
    service, mission = await _setup(tmp_path)
    try:
        dispatcher = PiToolDispatcher(service)
        lease = await _issue_lease(service, mission)
        path = tmp_path / "final.txt"
        path.write_bytes(b"completed artifact")
        final = await dispatcher.execute(
            _request(
                "rosclaw_deliver",
                mission=mission.mission_id,
                idem="finish_digest",
                lease=lease,
                arguments={"path": str(path), "role": "report"},
            )
        )
        assert final.ok, final.summary
        kernel = service._task_kernel
        task = kernel.latest_task_for(mission.mission_id, "pi_1")
        assert task["state"] == "SUCCEEDED"
        extra = tmp_path / "extra.txt"
        content = b"post-terminal supporting source"
        extra.write_bytes(content)
        appended = await dispatcher.execute(
            _request(
                "rosclaw_deliver",
                mission=mission.mission_id,
                idem="append_digest",
                lease=lease,
                arguments={"path": str(extra), "role": "diagnostic_source"},
            )
        )
        assert appended.ok, appended.summary
        assert f"sha256={hashlib.sha256(content).hexdigest()}" in appended.summary
        assert len(appended.artifact_refs) == 1
        row = kernel._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (appended.artifact_refs[0],),
        ).fetchone()
        assert row["sha256"] == hashlib.sha256(content).hexdigest()
        assert row["task_id"] == task["task_id"]
        assert kernel.get_task(task["task_id"])["state"] == "SUCCEEDED"
        assert appended.artifact_refs != final.artifact_refs
    finally:
        await service.close()
