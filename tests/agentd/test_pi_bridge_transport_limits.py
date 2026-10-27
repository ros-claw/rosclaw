"""Wire-byte regressions against the production AF_UNIX server, no model calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.operator_socket import MAX_REQUEST_BYTES
from rosclaw.agentd.pi_bridge.server import PiBridgeServer
from rosclaw.agentd.service import AgentService


@asynccontextmanager
async def bridge(tmp_path: Path):
    # Empty fixture gateway: any accidental model request fails, never succeeds.
    service = AgentService(
        load_agent_config(tmp_path / "config.yaml"),
        tmp_path,
        gateway=MockModelGateway(mock_profile(), []),
    )
    path = tmp_path / "run" / "pi-bridge.sock"
    server = PiBridgeServer(service, path)
    await server.start()
    try:
        yield service, path
    finally:
        await server.stop()
        await service.close()


def frame(token: str, size: int, unit: str = "x") -> bytes:
    # The token and session selector follow the padding: truncated input cannot
    # accidentally pass authentication or parse as a complete request.
    def encode(padding: str) -> bytes:
        return (json.dumps({
            "method": "pi.session.binding.get",
            "params": {"padding": padding, "pi_session_id": "absent", "token": token},
        }, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")

    available = size - len(encode(""))
    count, remainder = divmod(available, len(unit.encode("utf-8")))
    wire = encode(unit * count + "x" * remainder)
    assert len(wire) == size
    return wire


async def reply(reader: asyncio.StreamReader) -> dict:
    wire = await asyncio.wait_for(reader.readline(), 2)
    assert wire.endswith(b"\n"), "server silently closed instead of returning JSONL"
    assert len(wire) <= MAX_REQUEST_BYTES
    return json.loads(wire)


async def close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_wire_boundaries_utf8_fragmentation_and_repeated_connections(tmp_path: Path):
    assert MAX_REQUEST_BYTES == 256 * 1024
    async with bridge(tmp_path) as (service, path):
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
        for unit in ("x", "汉🙂e\u0301"):
            for size in (65535, 65536, 65537, 128 * 1024, MAX_REQUEST_BYTES - 1,
                         MAX_REQUEST_BYTES):
                wire = frame(service.control_token, size, unit)
                if unit != "x":
                    assert len(wire.decode("utf-8")) < len(wire)
                for fragmented in (False, True):
                    reader, writer = await asyncio.open_unix_connection(str(path))
                    try:
                        if fragmented:
                            # Deliberately split UTF8 codepoints as well as JSON tokens.
                            for start in range(0, len(wire), 997):
                                writer.write(wire[start:start + 997])
                                await writer.drain()
                                await asyncio.sleep(0)
                        else:
                            writer.write(wire)
                            await writer.drain()
                        assert await reply(reader) == {"ok": True, "binding": None}
                        # Same connection remains usable after the large request.
                        writer.write(frame(service.control_token, 256))
                        await writer.drain()
                        assert await reply(reader) == {"ok": True, "binding": None}
                    finally:
                        await close(writer)


@pytest.mark.asyncio
async def test_oversize_refusal_with_and_without_newline_then_recovery(tmp_path: Path):
    async with bridge(tmp_path) as (service, path):
        for terminated in (True, False):
            for size in (MAX_REQUEST_BYTES + 1, MAX_REQUEST_BYTES + 8192):
                wire = frame(service.control_token, size, "汉🙂")
                if not terminated:
                    # Still over cap; do not half-close or send a delimiter. The
                    # server must refuse without waiting for either event.
                    wire = wire[:-1] + b"x"
                reader, writer = await asyncio.open_unix_connection(str(path))
                try:
                    writer.write(wire)
                    await writer.drain()
                    assert await reply(reader) == {"ok": False, "error": "request too large"}
                    assert await asyncio.wait_for(reader.read(1), 2) == b""
                finally:
                    await close(writer)
                reader, writer = await asyncio.open_unix_connection(str(path))
                try:
                    writer.write(frame(service.control_token, MAX_REQUEST_BYTES))
                    await writer.drain()
                    assert (await reply(reader))["ok"] is True
                finally:
                    await close(writer)


@pytest.mark.asyncio
async def test_auth_malformed_recovery_and_kernel_peer_identity(tmp_path: Path):
    async with bridge(tmp_path) as (service, path):
        reader, writer = await asyncio.open_unix_connection(str(path))
        try:
            writer.write(frame("wrong-offline-token", 100 * 1024, "汉🙂"))
            await writer.drain()
            assert (await reply(reader))["code"] == "UNAUTHORIZED"
            for malformed in (b"{broken\n", b"[]\n", b"\xff\n"):
                writer.write(malformed)
                await writer.drain()
                assert (await reply(reader))["ok"] is False
            mission = service.create_mission("offline transport fixture")
            writer.write((json.dumps({"method": "pi.session.bind", "params": {
                "token": service.control_token, "mission_id": mission.mission_id,
                "pi_session_id": "offline-session", "caller_pid": -1, "caller_uid": -1,
            }}) + "\n").encode())
            await writer.drain()
            bound = await reply(reader)
            assert bound["ok"] is True
            assert bound["lease"]["owner_pid"] == os.getpid()
            assert bound["lease"]["owner_uid"] == os.getuid()
            writer.write(frame(service.control_token, 70000))
            await writer.drain()
            assert (await reply(reader))["ok"] is True
        finally:
            await close(writer)


@pytest.mark.asyncio
async def test_large_input_is_really_persisted_without_task_or_model(tmp_path: Path):
    async with bridge(tmp_path) as (service, path):
        mission = service.create_mission("offline input fixture")
        text = "汉🙂e\u0301" * 12000
        wire = (json.dumps({"method": "pi.input.persist", "params": {
            "token": service.control_token, "mission_id": mission.mission_id,
            "session_ref": "offline-session", "message_id": "offline-message", "text": text,
        }}, ensure_ascii=False) + "\n").encode("utf-8")
        assert 65536 < len(wire) <= MAX_REQUEST_BYTES
        reader, writer = await asyncio.open_unix_connection(str(path), limit=MAX_REQUEST_BYTES)
        try:
            writer.write(wire)
            await writer.drain()
            result = await reply(reader)
            assert result["ok"] is True
            assert result["input"]["input_id"]
            assert result["turn_disposition"]["suppress_model_turn"] is False
            persisted = [e for e in service.events_replay(mission.mission_id, limit=100)
                         if e.payload.get("message_id") == "offline-message"]
            assert len(persisted) == 1
            assert persisted[0].payload["bytes"] == len(text.encode("utf-8"))
            assert persisted[0].payload["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
            assert service._task_kernel.list_tasks(mission.mission_id) == []
        finally:
            await close(writer)
