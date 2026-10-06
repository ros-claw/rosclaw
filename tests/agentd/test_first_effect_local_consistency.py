"""Targeted tests: first-effect local admission consistency (P0-C repair).

Covers the server-side `pi.task.ensure_effect` entrypoint:
- missing/mismatched request context envelope is rejected BEFORE any
  Task/revision/binding mutation;
- invalid binding/writer (wrong session, wrong mission, foreign writer,
  expired lease) is rejected before mutation;
- a legitimate first effect creates exactly one task in the canonical
  session cwd, and a repeated effect does not bump the revision.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from rosclaw.agentd.operator_socket import operator_call
from tests.agentd.test_pi_bridge import _bridge

_TABLES = ["tasks", "task_revisions", "operations", "task_session_bindings"]


def _snapshot(service) -> dict:
    conn = service._store.connection
    return {
        table: [dict(row) for row in conn.execute("select * from " + table)] for table in _TABLES
    }


def _request_envelope(mission_id: str, session_id: str, **overrides) -> dict:
    envelope = {
        "schema_version": "rosclaw.pi_tool_request.v1",
        "request_id": "test_first_effect",
        "pi_session_id": session_id,
        "mission_id": mission_id,
        "context_revision": 0,
        "body_hash": "",
        "mode": "SIMULATION",
        "tool_name": "rosclaw_workspace_effect",
        "arguments": {},
        "requested_at": "2026-01-01T00:00:00+00:00",
        "idempotency_key": "test_first_effect",
        "actor": {"engine": "pi", "process_id": os.getpid(), "uid": os.getuid()},
    }
    envelope.update(overrides)
    return envelope


def _params(mission_id: str, session_id: str, cwd: str, **overrides) -> dict:
    params = {
        "mission_id": mission_id,
        "session_ref": session_id,
        "backend_native_id": session_id,
        "cwd": cwd,
        "mode": "SIMULATION",
        "request": _request_envelope(mission_id, session_id),
    }
    params.update(overrides)
    return params


@contextlib.contextmanager
def _sock_safe_base(tmp_path: Path) -> Iterator[Path]:
    # AF_UNIX 路径上限约 108 字节：按 UTF-8 编码后的字节长度判断，
    # 深层或含非 ASCII 字符的 basetmp 下 pi-bridge.sock 会超限。
    # 超限时在 /tmp 下创建短符号链接访问同一目录，保证 socket 路径足够短；
    # 该临时目录由本 helper 持有，退出时（含断言失败/异常）必定清理。
    sock_path = tmp_path / "run" / "pi-bridge.sock"
    if len(os.fsencode(sock_path)) <= 100:
        yield tmp_path
        return
    # 显式锚定 /tmp：TMPDIR 可能指向深层目录，默认 mkdtemp 位置仍可能超限。
    base = Path(tempfile.mkdtemp(prefix="felc_", dir="/tmp"))
    try:
        link = base / "b"
        link.symlink_to(tmp_path)
        yield link
    finally:
        shutil.rmtree(base, ignore_errors=True)


class TestEnsureEffectLocalConsistency:
    async def test_wrong_schema_rejected_before_mutation(self, tmp_path: Path) -> None:
        with _sock_safe_base(tmp_path) as base:
            service, server, sock = await _bridge(base)
            try:
                mission = service.create_mission("first-effect fixture")
                token = service.control_token
                bound = await operator_call(
                    sock,
                    "pi.session.bind",
                    {
                        "token": token,
                        "pi_session_id": "pi_s1",
                        "mission_id": mission.mission_id,
                    },
                )
                assert bound["ok"], bound
                service._task_kernel.persist_input(
                    mission_id=mission.mission_id,
                    session_ref="pi_s1",
                    message_id="m1",
                    text="fixture input",
                )
                service._store.connection.commit()
                before = _snapshot(service)
                # 显式错误 schema major 在 mutation 前 typed reject。
                denied = await operator_call(
                    sock,
                    "pi.task.ensure_effect",
                    {
                        "token": token,
                        **_params(
                            mission.mission_id,
                            "pi_s1",
                            str(tmp_path),
                            request=_request_envelope(
                                mission.mission_id,
                                "pi_s1",
                                schema_version="rosclaw.pi_tool_request.v999",
                            ),
                        ),
                    },
                )
                assert denied["ok"] is False and denied["code"] == "INVALID_REQUEST"
                # unknown-schema INVALID_REQUEST 之后、默认 schema 正向调用之前：
                # Task/revision/binding 快照必须保持零变更。
                assert _snapshot(service) == before
                # 缺省 schema_version 沿用 v1 默认——既有兼容保持正向。
                no_schema = _request_envelope(mission.mission_id, "pi_s1")
                del no_schema["schema_version"]
                accepted = await operator_call(
                    sock,
                    "pi.task.ensure_effect",
                    {
                        "token": token,
                        **_params(
                            mission.mission_id,
                            "pi_s1",
                            str(tmp_path),
                            request=no_schema,
                        ),
                    },
                )
                assert accepted["ok"] is True, accepted
            finally:
                await server.stop()
                await service.close()

    async def test_missing_request_context_rejected_before_mutation(self, tmp_path: Path) -> None:
        with _sock_safe_base(tmp_path) as base:
            service, server, sock = await _bridge(base)
            try:
                mission = service.create_mission("first-effect fixture")
                token = service.control_token
                bound = await operator_call(
                    sock,
                    "pi.session.bind",
                    {
                        "token": token,
                        "pi_session_id": "pi_s1",
                        "mission_id": mission.mission_id,
                    },
                )
                assert bound["ok"], bound
                service._task_kernel.persist_input(
                    mission_id=mission.mission_id,
                    session_ref="pi_s1",
                    message_id="m1",
                    text="fixture input",
                )
                service._store.connection.commit()
                before = _snapshot(service)
                params = _params(mission.mission_id, "pi_s1", str(tmp_path))
                del params["request"]
                denied = await operator_call(
                    sock, "pi.task.ensure_effect", {"token": token, **params}
                )
                assert denied["ok"] is False
                assert denied["code"] == "REQUEST_CONTEXT_REQUIRED"
                assert _snapshot(service) == before
            finally:
                await server.stop()
                await service.close()

    async def test_mismatched_request_context_rejected_before_mutation(
        self, tmp_path: Path
    ) -> None:
        with _sock_safe_base(tmp_path) as base:
            service, server, sock = await _bridge(base)
            try:
                mission = service.create_mission("first-effect fixture")
                other = service.create_mission("other fixture")
                token = service.control_token
                bound = await operator_call(
                    sock,
                    "pi.session.bind",
                    {
                        "token": token,
                        "pi_session_id": "pi_s1",
                        "mission_id": mission.mission_id,
                    },
                )
                assert bound["ok"], bound
                before = _snapshot(service)
                cases = [
                    _params(
                        mission.mission_id,
                        "pi_s1",
                        str(tmp_path),
                        request=_request_envelope(other.mission_id, "pi_s1"),
                    ),
                    _params(
                        mission.mission_id,
                        "pi_s1",
                        str(tmp_path),
                        request=_request_envelope(mission.mission_id, "pi_foreign"),
                    ),
                    _params(
                        mission.mission_id,
                        "pi_s1",
                        str(tmp_path),
                        request=_request_envelope(
                            mission.mission_id,
                            "pi_s1",
                            actor={
                                "engine": "pi",
                                "process_id": os.getpid() + 100000,
                                "uid": os.getuid(),
                            },
                        ),
                    ),
                ]
                expected = [
                    "REQUEST_MISSION_MISMATCH",
                    "REQUEST_SESSION_MISMATCH",
                    "REQUEST_ACTOR_MISMATCH",
                ]
                for params, code in zip(cases, expected, strict=True):
                    denied = await operator_call(
                        sock, "pi.task.ensure_effect", {"token": token, **params}
                    )
                    assert denied["ok"] is False, (code, denied)
                    assert denied["code"] == code, denied
                assert _snapshot(service) == before
            finally:
                await server.stop()
                await service.close()

    async def test_invalid_binding_or_writer_rejected_before_mutation(self, tmp_path: Path) -> None:
        with _sock_safe_base(tmp_path) as base:
            service, server, sock = await _bridge(base)
            try:
                mission = service.create_mission("first-effect fixture")
                other = service.create_mission("other fixture")
                token = service.control_token
                bound = await operator_call(
                    sock,
                    "pi.session.bind",
                    {
                        "token": token,
                        "pi_session_id": "pi_s1",
                        "mission_id": mission.mission_id,
                    },
                )
                assert bound["ok"], bound
                before = _snapshot(service)
                # 未绑定 session。
                denied = await operator_call(
                    sock,
                    "pi.task.ensure_effect",
                    {
                        "token": token,
                        **_params(mission.mission_id, "pi_unbound", str(tmp_path)),
                    },
                )
                assert denied["ok"] is False and denied["code"] == "SESSION_UNBOUND"
                # session 绑定的是别的 mission。
                denied = await operator_call(
                    sock,
                    "pi.task.ensure_effect",
                    {
                        "token": token,
                        **_params(other.mission_id, "pi_s1", str(tmp_path)),
                    },
                )
                assert denied["ok"] is False and denied["code"] == "MISSION_MISMATCH"
                # writer lease 过期。
                service._store.connection.execute(
                    "update pi_session_leases set expires_at='2000-01-01T00:00:00+00:00' "
                    "where mission_id=?",
                    (mission.mission_id,),
                )
                service._store.connection.commit()
                denied = await operator_call(
                    sock,
                    "pi.task.ensure_effect",
                    {
                        "token": token,
                        **_params(mission.mission_id, "pi_s1", str(tmp_path)),
                    },
                )
                assert denied["ok"] is False and denied["code"] == "WRITER_LEASE_REQUIRED"
                assert _snapshot(service) == before
            finally:
                await server.stop()
                await service.close()

    async def test_legitimate_first_effect_uses_canonical_cwd_and_no_rebump(
        self, tmp_path: Path
    ) -> None:
        with _sock_safe_base(tmp_path) as base:
            service, server, sock = await _bridge(base)
            try:
                mission = service.create_mission("first-effect fixture")
                token = service.control_token
                bound = await operator_call(
                    sock,
                    "pi.session.bind",
                    {
                        "token": token,
                        "pi_session_id": "pi_s1",
                        "mission_id": mission.mission_id,
                    },
                )
                assert bound["ok"], bound
                service._task_kernel.persist_input(
                    mission_id=mission.mission_id,
                    session_ref="pi_s1",
                    message_id="m1",
                    text="fixture input",
                )
                service._store.connection.commit()
                canonical = str(tmp_path / "session-workspace")
                first = await operator_call(
                    sock,
                    "pi.task.ensure_effect",
                    {
                        "token": token,
                        **_params(mission.mission_id, "pi_s1", canonical),
                    },
                )
                assert first["ok"] is True, first
                assert first["workspace_path"] == canonical
                conn = service._store.connection
                assert len(conn.execute("select * from tasks").fetchall()) == 1
                assert len(conn.execute("select * from task_revisions").fetchall()) == 1
                # 同一动机输入的连续 effectful call 不重复 bump revision。
                second = await operator_call(
                    sock,
                    "pi.task.ensure_effect",
                    {
                        "token": token,
                        **_params(mission.mission_id, "pi_s1", canonical),
                    },
                )
                assert second["ok"] is True, second
                assert second["task_id"] == first["task_id"]
                assert len(conn.execute("select * from task_revisions").fetchall()) == 1
            finally:
                await server.stop()
                await service.close()
