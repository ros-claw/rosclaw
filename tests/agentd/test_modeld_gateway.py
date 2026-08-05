"""批次 D：Python ModeldGateway 测试。

- 真实 modeld 子进程（Node >= 22.19 时）：UDS 权限、bearer 认证、
  无凭据诚实失败、崩溃诚实失败
- legacy backend 下 /model 热切换被拒绝并给出迁移指引
- MODEL_CONTROL 命令经 CommandService 路由（不发给模型）
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway, ModelGatewayError, ModelTurnRequest
from rosclaw.agentd.models.modeld_gateway import (
    _PROVIDER_MAP,
    ModeldGateway,
    _find_modeld_runtime,
)
from rosclaw.agentd.models.profiles import kimi_code_k3_profile, mock_profile
from rosclaw.agentd.service import AgentService

NODE_AVAILABLE = _find_modeld_runtime() is not None
requires_node = pytest.mark.skipif(not NODE_AVAILABLE, reason="node >= 22.19 or modeld not built")


def _request() -> ModelTurnRequest:
    return ModelTurnRequest(
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_output_tokens=16,
        mission_id="mis_x",
        context_id="ctx",
        context_revision=1,
    )


@requires_node
class TestModeldGateway:
    async def test_probe_without_credential_is_honest(self, tmp_path: Path) -> None:
        os.environ.pop("ROSCLAW_KIMI_API_KEY", None)
        os.environ.pop("KIMI_API_KEY", None)  # NA-FIX-7 broker 桥接键
        gateway = ModeldGateway(kimi_code_k3_profile(), home=tmp_path)
        try:
            probe = await gateway.probe()
            assert probe.error
            assert "no_credential" in probe.error or "missing" in probe.error.lower()
        finally:
            await gateway.close()

    async def test_stream_without_credential_fails_closed(self, tmp_path: Path) -> None:
        os.environ.pop("ROSCLAW_KIMI_API_KEY", None)
        os.environ.pop("KIMI_API_KEY", None)  # NA-FIX-7 broker 桥接键
        gateway = ModeldGateway(kimi_code_k3_profile(), home=tmp_path)
        try:
            with pytest.raises(ModelGatewayError, match="no_credential|missing"):
                await gateway.complete(_request())
        finally:
            await gateway.close()

    async def test_uds_permissions(self, tmp_path: Path) -> None:
        gateway = ModeldGateway(kimi_code_k3_profile(), home=tmp_path)
        try:
            await gateway._ensure_started()
            sock = Path(gateway._socket_path)
            assert sock.exists()
            assert (sock.stat().st_mode & 0o777) == 0o600
            assert (sock.parent.stat().st_mode & 0o700) == 0o700
        finally:
            await gateway.close()

    async def test_crash_is_honest(self, tmp_path: Path) -> None:
        gateway = ModeldGateway(kimi_code_k3_profile(), home=tmp_path)
        await gateway._ensure_started()
        # 杀死 modeld：下一次调用必须诚实报 modeld_crashed，不得伪造成功。
        gateway._proc.kill()
        gateway._proc.wait(timeout=5)
        with pytest.raises(ModelGatewayError, match="modeld_crashed"):
            await gateway.complete(_request())
        await gateway.close()

    async def test_management_providers_no_secrets(self, tmp_path: Path) -> None:
        gateway = ModeldGateway(kimi_code_k3_profile(), home=tmp_path)
        try:
            data = await gateway.manage("GET", "/v1/providers")
            ids = {p["id"] for p in data["providers"]}
            assert {"moonshot", "kimi-code", "ollama"} <= ids
            import json as _json

            assert "sk-" not in _json.dumps(data)
        finally:
            await gateway.close()

    async def test_unauthorized_management_rejected(self, tmp_path: Path) -> None:
        """无 token 直接打 modeld socket → 401（bearer 边界）。"""
        import aiohttp

        gateway = ModeldGateway(kimi_code_k3_profile(), home=tmp_path)
        try:
            await gateway._ensure_started()
            connector = aiohttp.UnixConnector(path=gateway._socket_path)
            async with (
                aiohttp.ClientSession(connector=connector) as session,
                session.get("http://localhost/v1/providers") as resp,
            ):
                assert resp.status == 401
        finally:
            await gateway.close()


class TestProviderMapping:
    def test_kimi_profiles_map(self) -> None:
        assert _PROVIDER_MAP["kimi_cn"] == "moonshot"
        assert _PROVIDER_MAP["kimi_code"] == "kimi-code"


class TestModelCommands:
    def _service(self, tmp_path: Path) -> AgentService:
        config = load_agent_config(tmp_path / "config.yaml")
        return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), []))

    async def test_legacy_backend_rejects_hot_switch(self, tmp_path: Path) -> None:
        service = self._service(tmp_path)
        try:
            result = service.switch_model("kimi-code", "k3")
            assert not result["ok"]
            assert result["error_code"] in ("legacy_backend", "no_profiles")
        finally:
            await service.close()

    @requires_node
    async def test_providers_command_routes_to_modeld(self, tmp_path: Path) -> None:
        from rosclaw.contracts.ui.commands import CommandRequestV1

        service = self._service(tmp_path)
        try:
            result = await service.commands.execute(
                CommandRequestV1(
                    request_id="r1",
                    idempotency_key="k1",
                    command_name="providers",
                    mission_id=None,
                )
            )
            assert result.ok
            ids = {p["id"] for p in result.data.get("providers", [])}
            assert "kimi-code" in ids
        finally:
            await service.close()

    @requires_node
    async def test_login_logout_commands(self, tmp_path: Path) -> None:
        from rosclaw.contracts.ui.commands import CommandRequestV1

        service = self._service(tmp_path)
        try:
            login = await service.commands.execute(
                CommandRequestV1(
                    request_id="r2",
                    idempotency_key="k2",
                    command_name="login",
                    arguments={"provider": "moonshot", "api_key": "sk-test-not-real"},
                    mission_id=None,
                )
            )
            assert login.ok
            assert "sk-test-not-real" not in login.message
            logout = await service.commands.execute(
                CommandRequestV1(
                    request_id="r3",
                    idempotency_key="k3",
                    command_name="logout",
                    arguments={"provider": "moonshot"},
                    mission_id=None,
                )
            )
            assert logout.ok
        finally:
            await service.close()

    async def test_login_requires_args(self, tmp_path: Path) -> None:
        from rosclaw.contracts.ui.commands import CommandRequestV1

        service = self._service(tmp_path)
        try:
            result = await service.commands.execute(
                CommandRequestV1(
                    request_id="r4",
                    idempotency_key="k4",
                    command_name="login",
                    arguments={},
                    mission_id=None,
                )
            )
            assert not result.ok and result.error_code == "invalid_arguments"
        finally:
            await service.close()
