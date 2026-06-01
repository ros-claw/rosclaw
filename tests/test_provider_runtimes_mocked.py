"""Tests for provider runtimes with mocked external deps."""

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock

from rosclaw.provider.runtimes.http_runtime import HTTPRuntime
from rosclaw.provider.runtimes.ros2_runtime import ROS2Runtime


class TestHTTPRuntimeMocked:
    @pytest.fixture(autouse=True)
    def cleanup_modules(self):
        yield
        sys.modules.pop("aiohttp", None)

    @pytest.mark.asyncio
    async def test_start_stop_with_mock_aiohttp(self):
        rt = HTTPRuntime("http_test", "http://localhost/api")
        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session
        sys.modules["aiohttp"] = mock_aiohttp
        await rt.start()
        assert rt._started is True
        await rt.stop()
        assert rt._started is False
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        rt = HTTPRuntime("http_test", "http://localhost/api")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"result": "ok"})
        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_session.close = AsyncMock()
        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session
        mock_aiohttp.ClientTimeout = MagicMock()
        sys.modules["aiohttp"] = mock_aiohttp
        await rt.start()
        result = await rt.invoke({"x": 1})
        assert result == {"result": "ok"}
        await rt.stop()

    @pytest.mark.asyncio
    async def test_invoke_http_error(self):
        rt = HTTPRuntime("http_test", "http://localhost/api")
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.json = AsyncMock(return_value={"error": "fail"})
        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_session.close = AsyncMock()
        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session
        mock_aiohttp.ClientTimeout = MagicMock()
        sys.modules["aiohttp"] = mock_aiohttp
        await rt.start()
        from rosclaw.provider.core.errors import RuntimeAdapterError
        with pytest.raises(RuntimeAdapterError, match="HTTP 500"):
            await rt.invoke({"x": 1})
        await rt.stop()

    @pytest.mark.asyncio
    async def test_invoke_retry_then_success(self):
        rt = HTTPRuntime("http_test", "http://localhost/api", retries=2)
        bad_resp = MagicMock()
        bad_resp.status = 500
        bad_resp.json = AsyncMock(return_value={"error": "fail"})
        good_resp = MagicMock()
        good_resp.status = 200
        good_resp.json = AsyncMock(return_value={"result": "ok"})
        call_count = 0
        def fake_post(*args, **kwargs):  # noqa: E306
            nonlocal call_count
            call_count += 1
            resp = good_resp if call_count > 1 else bad_resp
            return MagicMock(__aenter__=AsyncMock(return_value=resp), __aexit__=AsyncMock(return_value=False))
        mock_session = MagicMock()
        mock_session.post = fake_post
        mock_session.close = AsyncMock()
        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session
        mock_aiohttp.ClientTimeout = MagicMock()
        sys.modules["aiohttp"] = mock_aiohttp
        await rt.start()
        result = await rt.invoke({"x": 1})
        assert result == {"result": "ok"}
        assert call_count == 2
        await rt.stop()


class TestROS2RuntimeMocked:
    @pytest.fixture(autouse=True)
    def cleanup_modules(self):
        yield
        for mod in ["rclpy", "rclpy.node"]:
            sys.modules.pop(mod, None)

    @pytest.mark.asyncio
    async def test_start_stop_with_mock_rclpy(self):
        rt = ROS2Runtime("ros2_test", action_name="/move", service_name="/srv")
        mock_node = MagicMock()
        mock_rclpy = MagicMock()
        mock_rclpy.ok.return_value = True
        mock_Node = MagicMock(return_value=mock_node)
        mock_rclpy_module = MagicMock()
        mock_rclpy_module.Node = mock_Node
        sys.modules["rclpy"] = mock_rclpy_module
        sys.modules["rclpy.node"] = mock_rclpy_module
        await rt.start()
        assert rt._started is True
        assert rt._node is mock_node
        await rt.stop()
        assert rt._started is False
        mock_node.destroy_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_rclpy_not_ok(self):
        rt = ROS2Runtime("ros2_test")
        mock_node = MagicMock()
        mock_Node = MagicMock(return_value=mock_node)
        mock_rclpy_module = MagicMock()
        mock_rclpy_module.ok.return_value = False
        mock_rclpy_module.Node = mock_Node
        sys.modules["rclpy"] = mock_rclpy_module
        sys.modules["rclpy.node"] = mock_rclpy_module
        await rt.start()
        mock_rclpy_module.init.assert_called_once()
        assert rt._started is True
        await rt.stop()
