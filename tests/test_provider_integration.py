"""Integration tests for rosclaw.provider + Runtime.

These tests verify that the Provider system integrates correctly with
Runtime lifecycle, including sync/async boundary handling.
"""

import asyncio

import pytest

from rosclaw.provider.core import (
    CapabilityRouter,
    Provider,
    ProviderManifest,
    ProviderRegistry,
    ProviderRequest,
    ProviderResponse,
)


class DummyProvider(Provider):
    """Test provider that returns a simple result."""

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={"output": "ok"},
        )

    async def load(self) -> None:
        pass

    async def health(self) -> dict:
        return {"ok": True, "provider": self.name}


class AsyncLoadProvider(Provider):
    """Provider whose load() method does async work."""

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={"output": "async_loaded"},
        )

    async def load(self) -> None:
        await asyncio.sleep(0.01)

    async def health(self) -> dict:
        return {"ok": True, "provider": self.name}


# ---------------------------------------------------------------------------
# Sync context registration
# ---------------------------------------------------------------------------

class TestSyncContextRegistration:
    @pytest.mark.asyncio
    async def test_register_sync_context_auto_load(self):
        """register() from sync context with auto_load=True should work."""
        reg = ProviderRegistry()
        manifest = ProviderManifest(
            name="sync_provider", version="1.0", type="llm"
        )
        p = reg.register(manifest, lambda m: DummyProvider(m), auto_load=True)
        # Allow deferred load coroutine to complete
        await asyncio.sleep(0.01)
        assert p._healthy is True
        assert reg.is_healthy("sync_provider") is True

    def test_register_sync_context_no_auto_load(self):
        """register() from sync context with auto_load=False should not load."""
        reg = ProviderRegistry()
        manifest = ProviderManifest(
            name="no_load", version="1.0", type="llm"
        )
        p = reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        assert p._healthy is False  # not loaded
        assert reg.is_healthy("no_load") is False

    @pytest.mark.asyncio
    async def test_unregister_sync_context(self):
        """unregister() from sync context should work."""
        reg = ProviderRegistry()
        manifest = ProviderManifest(name="to_remove", version="1.0", type="llm")
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        reg.unregister("to_remove")
        # Allow unload coroutine to complete
        await asyncio.sleep(0.01)
        assert reg.list_providers() == []


# ---------------------------------------------------------------------------
# Async context registration
# ---------------------------------------------------------------------------

class TestAsyncContextRegistration:
    @pytest.mark.asyncio
    async def test_register_async_context_auto_load(self):
        """register() from async context with auto_load=True should not crash."""
        reg = ProviderRegistry()
        manifest = ProviderManifest(
            name="async_provider", version="1.0", type="llm"
        )
        p = reg.register(manifest, lambda m: DummyProvider(m), auto_load=True)
        # Deferred load is scheduled; wait for it
        await asyncio.sleep(0.05)
        assert p._healthy is True
        assert reg.is_healthy("async_provider") is True

    @pytest.mark.asyncio
    async def test_register_async_context_with_async_load(self):
        """register() with a provider that does real async work in load()."""
        reg = ProviderRegistry()
        manifest = ProviderManifest(
            name="async_load_provider", version="1.0", type="llm"
        )
        p = reg.register(manifest, lambda m: AsyncLoadProvider(m), auto_load=True)
        # Deferred load includes asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert p._healthy is True

    @pytest.mark.asyncio
    async def test_unregister_async_context(self):
        """unregister() from async context should not crash."""
        reg = ProviderRegistry()
        manifest = ProviderManifest(name="async_remove", version="1.0", type="llm")
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        reg.unregister("async_remove")
        assert reg.list_providers() == []


# ---------------------------------------------------------------------------
# ProviderRegistry.set_provider_health
# ---------------------------------------------------------------------------

class TestSetProviderHealth:
    def test_set_health_public_api(self):
        """set_provider_health() should update both provider and registry state."""
        reg = ProviderRegistry()
        manifest = ProviderManifest(name="mock", version="1.0", type="llm")
        p = reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        assert p._healthy is False
        assert reg.is_healthy("mock") is False

        reg.set_provider_health("mock", ok=True)
        assert p._healthy is True
        assert reg.is_healthy("mock") is True

    def test_set_health_with_error(self):
        """set_provider_health() with error should store it."""
        reg = ProviderRegistry()
        manifest = ProviderManifest(name="fail", version="1.0", type="llm")
        p = reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        reg.set_provider_health("fail", ok=False, error="load timeout")
        assert p._healthy is False
        assert p._load_error == "load timeout"
        assert reg.is_healthy("fail") is False


# ---------------------------------------------------------------------------
# CapabilityRouter integration
# ---------------------------------------------------------------------------

class TestCapabilityRouterIntegration:
    @pytest.fixture
    def router(self) -> CapabilityRouter:
        reg = ProviderRegistry()
        manifest = ProviderManifest(
            name="llm_provider",
            version="1.0",
            type="llm",
            capabilities=["llm.chat"],
        )
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        reg.set_provider_health("llm_provider", ok=True)
        return CapabilityRouter(reg)

    @pytest.mark.asyncio
    async def test_invoke_through_router(self, router: CapabilityRouter):
        req = ProviderRequest(
            request_id="r1", capability="llm.chat", inputs={"text": "hi"}
        )
        resp = await router.invoke(req)
        assert resp.is_ok
        assert resp.provider == "llm_provider"
        assert resp.result["output"] == "ok"

    @pytest.mark.asyncio
    async def test_route_decision(self, router: CapabilityRouter):
        req = ProviderRequest(
            request_id="r1", capability="llm.chat", inputs={"text": "hi"}
        )
        decision = await router.route(req)
        assert decision.selected_provider == "llm_provider"
        assert decision.score > 0


# ---------------------------------------------------------------------------
# End-to-end: Runtime with providers
# ---------------------------------------------------------------------------

class TestRuntimeWithProviders:
    @pytest.mark.asyncio
    async def test_runtime_builtin_providers(self):
        """Verify Runtime initializes its built-in mock providers correctly."""
        from rosclaw.core import Runtime, RuntimeConfig

        runtime = Runtime(RuntimeConfig())
        runtime.initialize()

        registry = runtime.provider_registry
        assert registry is not None
        assert "mock_vlm" in registry.list_providers()
        assert "mock_skill" in registry.list_providers()
        assert "mock_critic" in registry.list_providers()

        # All built-in providers should be marked healthy via set_provider_health
        assert registry.is_healthy("mock_vlm") is True
        assert registry.is_healthy("mock_skill") is True
        assert registry.is_healthy("mock_critic") is True

        router = runtime.capability_router
        assert router is not None

        # Invoke a VLM capability through the router
        req = ProviderRequest(
            request_id="r1",
            capability="vlm.object_grounding",
            inputs={"query": "cup"},
        )
        resp = await router.invoke(req)
        assert resp.is_ok
        assert "objects" in resp.result

        runtime.stop()

    @pytest.mark.asyncio
    async def test_runtime_register_and_invoke_custom_provider(self):
        """Register a custom provider via Runtime and invoke it."""
        from rosclaw.core import Runtime, RuntimeConfig

        runtime = Runtime(RuntimeConfig())
        runtime.initialize()

        manifest = ProviderManifest(
            name="custom_llm",
            version="1.0",
            type="llm",
            capabilities=["llm.chat"],
        )
        runtime.provider_registry.register(
            manifest, lambda m: DummyProvider(m), auto_load=False
        )
        runtime.provider_registry.set_provider_health("custom_llm", ok=True)

        # Ensure deepseek does not steal the route when env key is present
        if "deepseek" in runtime.provider_registry.list_providers():
            runtime.provider_registry.set_provider_health("deepseek", ok=False)

        req = ProviderRequest(
            request_id="r1", capability="llm.chat", inputs={"text": "hi"}
        )
        resp = await runtime.capability_router.invoke(req)
        assert resp.is_ok
        assert resp.provider == "custom_llm"

        runtime.stop()
