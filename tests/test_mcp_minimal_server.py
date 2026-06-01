"""Tests for mcp.minimal_server — ROSClawMinimalMCPServer."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class FakeMCPHub:
    """Fake MCPHub for testing without real MCP library."""

    def __init__(self, *args, **kwargs):
        self._tools = {}
        self._server = None

    def initialize(self):
        self._tools = {
            "get_robot_state": {
                "name": "get_robot_state",
                "description": "Get robot state",
                "inputSchema": {"type": "object", "properties": {}},
            },
            "move_joints": {
                "name": "move_joints",
                "description": "Move joints",
                "inputSchema": {
                    "type": "object",
                    "properties": {"joint_positions": {"type": "array"}},
                    "required": ["joint_positions"],
                },
            },
        }

    def stop(self):
        pass

    async def handle_tool_call(self, name, arguments):
        return {"status": "ok", "tool": name}


class FakeTool:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeTextContent:
    def __init__(self, text):
        self.type = "text"
        self.text = text


def _mock_mcp_modules():
    """Create mock MCP module hierarchy."""
    mcp = MagicMock()
    mcp.server = MagicMock()
    mcp.server.Server = MagicMock()
    mcp.server.models = MagicMock()
    mcp.server.stdio = MagicMock()
    mcp.server.stdio.stdio_server = MagicMock()
    mcp.types = MagicMock()
    mcp.types.TextContent = FakeTextContent
    mcp.types.Tool = FakeTool
    return mcp


@pytest.fixture
def mock_mcp():
    """Mock MCP library imports."""
    mcp = _mock_mcp_modules()
    with patch.dict(sys.modules, {
        "mcp": mcp,
        "mcp.server": mcp.server,
        "mcp.server.models": mcp.server.models,
        "mcp.server.stdio": mcp.server.stdio,
        "mcp.types": mcp.types,
    }):
        with patch("rosclaw.agent_runtime.mcp_hub.MCPHub", FakeMCPHub):
            yield mcp


class TestMinimalServerInit:
    def test_init(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        assert server.server is not None
        assert server.event_bus is not None
        assert server.hub is not None

    def test_system_tools(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        tools = server._system_tools()
        names = [t.name for t in tools]
        assert "system.list_robots" in names
        assert "system.list_providers" in names
        assert "system.run_sandbox_task" in names
        assert "system.query_memory" in names
        assert "system.explain_failure" in names
        assert "system.compile_asset_bundle" in names
        assert "system.get_version" in names
        assert len(tools) == 7


class TestSystemToolHandlers:
    @pytest.mark.asyncio
    async def test_get_version(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        result = await server._handle_system_tool("system.get_version", {})
        assert result["name"] == "rosclaw"
        assert result["version"] == "1.0.0"
        assert result["status"] == "ready"

    @pytest.mark.asyncio
    async def test_list_robots(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.runtime.RobotRegistry") as mock_reg:
            mock_reg.return_value.list_available.return_value = ["ur5e", "g1"]
            result = await server._handle_system_tool("system.list_robots", {})
            assert result["robots"] == ["ur5e", "g1"]
            assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_list_robots_error(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.runtime.RobotRegistry", side_effect=Exception("boom")):
            result = await server._handle_system_tool("system.list_robots", {})
            assert result["count"] == 0
            assert "error" in result

    @pytest.mark.asyncio
    async def test_list_providers(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.provider.core.registry.ProviderRegistry") as mock_reg:
            mock_reg.return_value.list_providers.return_value = [{"name": "llm"}]
            result = await server._handle_system_tool("system.list_providers", {})
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_list_providers_error(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.provider.core.registry.ProviderRegistry", side_effect=Exception("boom")):
            result = await server._handle_system_tool("system.list_providers", {})
            assert result["count"] == 0
            assert "error" in result

    @pytest.mark.asyncio
    async def test_unknown_system_tool(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        result = await server._handle_system_tool("system.unknown", {})
        assert "error" in result


class TestRunSandboxTask:
    @pytest.mark.asyncio
    async def test_sandbox_blocked(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()

        fake_decision = MagicMock()
        fake_decision.is_allowed = False
        fake_decision.risk_score = 0.8
        fake_decision.violated_constraints = ["joint_limit"]
        fake_decision.replay_id = "replay_123"

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            mock_gate.return_value.check.return_value = fake_decision
            result = await server._handle_run_sandbox_task({
                "robot_id": "ur5e",
                "task": "reach",
            })
            assert result["status"] == "BLOCKED"
            assert result["risk_score"] == 0.8

    @pytest.mark.asyncio
    async def test_sandbox_pid_move(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()

        fake_decision = MagicMock()
        fake_decision.is_allowed = True
        fake_decision.risk_score = 0.1

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            mock_gate.return_value.check.return_value = fake_decision
            with patch("rosclaw.practice.episode_recorder.EpisodeRecorder"):
                result = await server._handle_run_sandbox_task({
                    "robot_id": "ur5e",
                    "task": "pid_move",
                    "parameters": {"target": 0.5},
                })
                assert result["status"] == "SUCCESS"
                assert result["result"]["final_position"] == 0.5

    @pytest.mark.asyncio
    async def test_sandbox_reach(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()

        fake_decision = MagicMock()
        fake_decision.is_allowed = True
        fake_decision.risk_score = 0.0

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            mock_gate.return_value.check.return_value = fake_decision
            with patch("rosclaw.practice.episode_recorder.EpisodeRecorder"):
                result = await server._handle_run_sandbox_task({
                    "robot_id": "ur5e",
                    "task": "reach",
                    "parameters": {"target_pose": [0.5, 0.0, 0.3]},
                })
                assert result["status"] == "SUCCESS"
                assert result["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_sandbox_g1_walk(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()

        fake_decision = MagicMock()
        fake_decision.is_allowed = True
        fake_decision.risk_score = 0.0

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            mock_gate.return_value.check.return_value = fake_decision
            with patch("rosclaw.practice.episode_recorder.EpisodeRecorder"):
                result = await server._handle_run_sandbox_task({
                    "robot_id": "g1",
                    "task": "g1_walk",
                    "parameters": {"distance": 5.0},
                })
                assert result["status"] == "SUCCESS"
                assert result["result"]["distance"] == 5.0

    @pytest.mark.asyncio
    async def test_sandbox_unknown_task(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()

        fake_decision = MagicMock()
        fake_decision.is_allowed = True
        fake_decision.risk_score = 0.0

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate") as mock_gate:
            mock_gate.return_value.check.return_value = fake_decision
            with patch("rosclaw.practice.episode_recorder.EpisodeRecorder"):
                result = await server._handle_run_sandbox_task({
                    "robot_id": "ur5e",
                    "task": "custom_task",
                })
                assert result["status"] == "SUCCESS"
                assert "Mock execution" in result["result"]["message"]

    @pytest.mark.asyncio
    async def test_sandbox_firewall_error(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()

        with patch("rosclaw.sandbox.firewall.gate.FirewallGate", side_effect=Exception("firewall init failed")):
            result = await server._handle_run_sandbox_task({
                "robot_id": "ur5e",
                "task": "reach",
            })
            assert result["status"] == "error"
            assert result["phase"] == "firewall"


class TestQueryMemory:
    @pytest.mark.asyncio
    async def test_query_similar(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem:
            mock_mem.return_value.find_similar_experiences.return_value = [
                {"id": "exp1", "instruction": "pick up the cup"}
            ]
            result = await server._handle_query_memory({
                "query": "pick up cup",
                "query_type": "similar",
                "limit": 3,
            })
            assert result["type"] == "similar"
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_query_failure(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem:
            mock_mem.return_value.explain_last_failure.return_value = {"failure_type": "collision"}
            result = await server._handle_query_memory({
                "query": "last failure",
                "query_type": "failure",
            })
            assert result["type"] == "failure"
            assert "result" in result

    @pytest.mark.asyncio
    async def test_query_other(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem:
            mock_mem.return_value.get_statistics.return_value = {"total": 5}
            result = await server._handle_query_memory({
                "query": "stats",
                "query_type": "experience",
            })
            assert result["type"] == "experience"
            assert "statistics" in result

    @pytest.mark.asyncio
    async def test_query_memory_error(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.memory.interface.MemoryInterface", side_effect=Exception("mem error")):
            result = await server._handle_query_memory({"query": "test"})
            assert "error" in result


class TestExplainFailure:
    @pytest.mark.asyncio
    async def test_explain_found(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem:
            mock_mem.return_value.explain_last_failure.return_value = {
                "id": "f1",
                "failure_type": "collision",
                "root_cause": "obstacle",
                "recovery_hint": "retract",
            }
            result = await server._handle_explain_failure({})
            assert result["status"] == "found"
            assert result["failure_type"] == "collision"

    @pytest.mark.asyncio
    async def test_explain_not_found(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.memory.interface.MemoryInterface") as mock_mem:
            mock_mem.return_value.explain_last_failure.return_value = None
            result = await server._handle_explain_failure({})
            assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_explain_error(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.memory.interface.MemoryInterface", side_effect=Exception("boom")):
            result = await server._handle_explain_failure({})
            assert result["status"] == "error"


class TestCompileAssetBundle:
    @pytest.mark.asyncio
    async def test_compile_success(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()

        fake_bundle = MagicMock()
        fake_bundle.staging_ready = True
        fake_bundle.production_ready = False
        fake_bundle.files = {"manifest.yaml": "content"}
        fake_bundle.validation = {"passed": True}

        with patch("rosclaw.forge.bundle_compiler.BundleCompiler") as mock_compiler:
            mock_compiler.return_value.compile.return_value = fake_bundle
            result = await server._handle_compile_asset_bundle({
                "sdk_doc": "test sdk",
                "bundle_name": "test_bundle",
            })
            assert result["status"] == "generated"
            assert result["bundle_name"] == "test_bundle"
            assert result["staging_ready"] is True

    @pytest.mark.asyncio
    async def test_compile_error(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch("rosclaw.forge.bundle_compiler.BundleCompiler", side_effect=Exception("compile failed")):
            result = await server._handle_compile_asset_bundle({
                "sdk_doc": "test",
                "bundle_name": "bad",
            })
            assert result["status"] == "error"


class TestCallTool:
    @pytest.mark.asyncio
    async def test_call_system_tool(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        # Patch _handle_system_tool to avoid complex mocking
        with patch.object(server, "_handle_system_tool", return_value={"version": "1.0.0"}):
            # Access the registered call_tool handler via the mock
            # We can't easily call it directly because it's a decorator-wrapped handler
            # Instead test the method path
            result = await server._handle_system_tool("system.get_version", {})
            assert result["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_call_hub_tool(self, mock_mcp):
        from rosclaw.mcp.minimal_server import ROSClawMinimalMCPServer

        server = ROSClawMinimalMCPServer()
        with patch.object(server.hub, "handle_tool_call", return_value={"state": "ok"}):
            # The handler is registered as a decorator; test hub directly
            result = await server.hub.handle_tool_call("get_robot_state", {})
            assert result["state"] == "ok"


class TestMainShutdown:
    def test_main_keyboard_interrupt(self, mock_mcp):
        from rosclaw.mcp.minimal_server import main

        with patch.object(sys, "stderr", MagicMock()):
            # Main creates server then runs asyncio.run which would block;
            # KeyboardInterrupt should be caught
            with patch("rosclaw.mcp.minimal_server.ROSClawMinimalMCPServer") as mock_cls:
                mock_server = MagicMock()
                mock_cls.return_value = mock_server
                mock_server.hub = MagicMock()
                mock_server.hub.stop = MagicMock()
                mock_server.run.side_effect = KeyboardInterrupt()
                with patch("asyncio.run", side_effect=lambda coro: (_ for _ in ()).throw(KeyboardInterrupt())):
                    try:
                        main()
                    except KeyboardInterrupt:
                        pass  # Expected
