"""Tests for provider runtime adapters."""

import pytest

from rosclaw.provider.runtimes.base import RuntimeAdapter
from rosclaw.provider.runtimes.http_runtime import HTTPRuntime
from rosclaw.provider.runtimes.python_runtime import PythonRuntime
from rosclaw.provider.runtimes.ros2_runtime import ROS2Runtime
from rosclaw.provider.core.errors import RuntimeAdapterError


class TestRuntimeAdapterBase:
    def test_init_defaults(self):
        class Dummy(RuntimeAdapter):
            async def start(self): ...  # noqa: E704
            async def stop(self): ...  # noqa: E704
            async def invoke(self, payload): ...  # noqa: E704

        d = Dummy("test")
        assert d.name == "test"
        assert d.config == {}
        assert d._started is False

    def test_init_with_config(self):
        class Dummy(RuntimeAdapter):
            async def start(self): ...  # noqa: E704
            async def stop(self): ...  # noqa: E704
            async def invoke(self, payload): ...  # noqa: E704

        d = Dummy("test", {"key": "val"})
        assert d.config == {"key": "val"}

    def test_ensure_started_raises(self):
        class Dummy(RuntimeAdapter):
            async def start(self): ...  # noqa: E704
            async def stop(self): ...  # noqa: E704
            async def invoke(self, payload): ...  # noqa: E704

        d = Dummy("test")
        with pytest.raises(RuntimeError, match="not started"):
            d.ensure_started()


class TestHTTPRuntime:
    @pytest.mark.asyncio
    async def test_init(self):
        rt = HTTPRuntime("http_test", "http://localhost:8080/api", timeout_sec=5.0, retries=2, headers={"X-Key": "v"})
        assert rt.endpoint == "http://localhost:8080/api"
        assert rt.timeout_sec == 5.0
        assert rt.retries == 2
        assert rt.headers == {"X-Key": "v"}
        assert rt._session is None

    @pytest.mark.asyncio
    async def test_start_stop_without_aiohttp(self, monkeypatch):
        """start raises when aiohttp is missing."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "aiohttp":
                raise ImportError("no aiohttp")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rt = HTTPRuntime("http_test", "http://localhost")
        with pytest.raises(RuntimeError, match="aiohttp is required"):
            await rt.start()

    @pytest.mark.asyncio
    async def test_invoke_not_started(self):
        rt = HTTPRuntime("http_test", "http://localhost")
        with pytest.raises(RuntimeError, match="not started"):
            await rt.invoke({"x": 1})

    @pytest.mark.asyncio
    async def test_invoke_session_none(self):
        rt = HTTPRuntime("http_test", "http://localhost")
        rt._started = True
        with pytest.raises(RuntimeAdapterError, match="Session not initialized"):
            await rt.invoke({"x": 1})


class TestPythonRuntime:
    @pytest.mark.asyncio
    async def test_init_and_bind(self):
        def fn(payload):
            return {"result": payload["x"] * 2}

        rt = PythonRuntime("py_test", fn=fn)
        assert rt._fn is fn

    @pytest.mark.asyncio
    async def test_bind_after_init(self):
        rt = PythonRuntime("py_test")
        assert rt._fn is None
        def fn(p): return p  # noqa: E704
        rt.bind(fn)
        assert rt._fn is fn

    @pytest.mark.asyncio
    async def test_start_no_callable(self):
        rt = PythonRuntime("py_test")
        with pytest.raises(RuntimeAdapterError, match="No callable bound"):
            await rt.start()

    @pytest.mark.asyncio
    async def test_start_and_invoke(self):
        def fn(payload):
            return {"out": payload["in"] + 1}

        rt = PythonRuntime("py_test", fn=fn)
        await rt.start()
        assert rt._started is True
        result = await rt.invoke({"in": 5})
        assert result == {"out": 6}
        await rt.stop()
        assert rt._started is False

    @pytest.mark.asyncio
    async def test_invoke_not_started(self):
        def fn(p): return p  # noqa: E704
        rt = PythonRuntime("py_test", fn=fn)
        with pytest.raises(RuntimeError, match="not started"):
            await rt.invoke({"x": 1})

    @pytest.mark.asyncio
    async def test_invoke_fn_none(self):
        rt = PythonRuntime("py_test")
        rt._started = True
        with pytest.raises(RuntimeAdapterError, match="No callable bound"):
            await rt.invoke({"x": 1})

    @pytest.mark.asyncio
    async def test_invoke_exception(self):
        def bad_fn(payload):
            raise ValueError("boom")

        rt = PythonRuntime("py_test", fn=bad_fn)
        await rt.start()
        with pytest.raises(RuntimeAdapterError, match="boom"):
            await rt.invoke({"x": 1})


class TestROS2Runtime:
    @pytest.mark.asyncio
    async def test_init(self):
        rt = ROS2Runtime("ros2_test", action_name="/move_action", service_name="/srv", timeout_sec=10.0)
        assert rt.action_name == "/move_action"
        assert rt.service_name == "/srv"
        assert rt.timeout_sec == 10.0
        assert rt._node is None
        assert rt._client is None

    @pytest.mark.asyncio
    async def test_start_without_rclpy(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "rclpy":
                raise ImportError("no rclpy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rt = ROS2Runtime("ros2_test")
        with pytest.raises(RuntimeError, match="rclpy is required"):
            await rt.start()

    @pytest.mark.asyncio
    async def test_stop_no_node(self):
        rt = ROS2Runtime("ros2_test")
        await rt.stop()
        assert rt._started is False

    @pytest.mark.asyncio
    async def test_invoke_not_implemented(self):
        rt = ROS2Runtime("ros2_test")
        rt._started = True
        with pytest.raises(RuntimeAdapterError, match="not yet implemented"):
            await rt.invoke({"x": 1})
