"""Constructor-only SIM MCP environment contract; no child/model/robot starts."""

import pytest

from rosclaw.agentd.config import AgentConfig
from rosclaw.agentd.service import AgentService


@pytest.mark.asyncio
async def test_persistent_sim_channel_honors_explicit_installation_env_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/explicit/public/capability")
    monkeypatch.setenv("HOMECLEAN_UNDECLARED_SECRET_FIXTURE", "fixture-only-not-a-real-secret")
    monkeypatch.delenv("ROSCLAW_DAEMON_SOCKET", raising=False)
    config = AgentConfig(
        enabled=True,
        sim_body_id="sim/ur5e",
        mcp_servers=[
            {
                "name": "explicit-fixture",
                "command": "/not-started",
                "args": [],
                "sim_executor": True,
                "env_refs": ["PYTHONPATH"],
            }
        ],
        raw={},
    )
    service = AgentService(config, tmp_path)
    try:
        channel = service._sim_executors["mcp:explicit-fixture"]
        adapter = next(a for a in service._mcp_adapters if a.source == "mcp:explicit-fixture")
        assert channel._client is adapter._client
        assert channel._client._env.get("PYTHONPATH") == "/explicit/public/capability"
        assert "HOMECLEAN_UNDECLARED_SECRET_FIXTURE" not in channel._client._env
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_persistent_sim_channel_does_not_inherit_undeclared_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/undeclared/private/fixture")
    monkeypatch.delenv("ROSCLAW_DAEMON_SOCKET", raising=False)
    config = AgentConfig(
        enabled=True,
        sim_body_id="sim/ur5e",
        mcp_servers=[
            {"name": "no-refs", "command": "/not-started", "args": [], "sim_executor": True}
        ],
        raw={},
    )
    service = AgentService(config, tmp_path)
    try:
        assert "PYTHONPATH" not in service._sim_executors["mcp:no-refs"]._client._env
    finally:
        await service.close()
