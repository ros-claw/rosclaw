"""Provider coverage tests — fills gaps in generic adapter, loader, router."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from rosclaw.provider.adapters.generic import GenericProvider
from rosclaw.provider.core.errors import ProviderNotFoundError, RuntimeAdapterError
from rosclaw.provider.core.manifest import ProviderManifest, RuntimeSpec, SafetySpec
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.loader import ProviderLoader
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.router import CapabilityRouter


# --- GenericProvider ---

class TestGenericProviderCoverage:
    def test_parse_headers_multiline(self):
        headers = GenericProvider._parse_headers("Auth: token\nContent-Type: json")
        assert headers == {"Auth": "token", "Content-Type": "json"}

    def test_parse_headers_empty(self):
        assert GenericProvider._parse_headers("") == {}

    def test_parse_headers_no_colon(self):
        assert GenericProvider._parse_headers("nocolon") == {}

    def test_create_runtime_unsupported_backend(self):
        manifest = ProviderManifest.from_dict({
            "name": "bad_backend",
            "version": "0.1.0",
            "type": "vlm",
            "capabilities": ["vlm.x"],
            "runtime": {"backend": "quantum"},
        })
        with pytest.raises(RuntimeAdapterError, match="Unsupported backend"):
            GenericProvider(manifest)

    @pytest.mark.asyncio
    async def test_infer_no_runtime(self):
        manifest = ProviderManifest.from_dict({
            "name": "meta",
            "version": "0.1.0",
            "type": "critic",
            "capabilities": ["critic.x"],
        })
        provider = GenericProvider(manifest)
        req = ProviderRequest(request_id="r1", capability="critic.x", inputs={})
        with pytest.raises(RuntimeAdapterError, match="No runtime adapter"):
            await provider.infer(req)

    @pytest.mark.asyncio
    async def test_infer_runtime_adapter_error_propagates(self):
        manifest = ProviderManifest.from_dict({
            "name": "http_err",
            "version": "0.1.0",
            "type": "vlm",
            "capabilities": ["vlm.x"],
            "runtime": {"backend": "http", "endpoint": "http://x"},
        })
        provider = GenericProvider(manifest)
        req = ProviderRequest(request_id="r1", capability="vlm.x", inputs={})

        mock_runtime = MagicMock()
        mock_runtime.invoke = AsyncMock(side_effect=RuntimeAdapterError("boom"))
        provider._runtime = mock_runtime

        with pytest.raises(RuntimeAdapterError, match="boom"):
            await provider.infer(req)

    @pytest.mark.asyncio
    async def test_infer_generic_exception_wrapped(self):
        manifest = ProviderManifest.from_dict({
            "name": "http_wrap",
            "version": "0.1.0",
            "type": "vlm",
            "capabilities": ["vlm.x"],
            "runtime": {"backend": "http", "endpoint": "http://x"},
        })
        provider = GenericProvider(manifest)
        req = ProviderRequest(request_id="r1", capability="vlm.x", inputs={})

        mock_runtime = MagicMock()
        mock_runtime.invoke = AsyncMock(side_effect=ValueError("raw error"))
        provider._runtime = mock_runtime

        with pytest.raises(RuntimeAdapterError, match="Runtime invoke failed"):
            await provider.infer(req)

    @pytest.mark.asyncio
    async def test_infer_result_parsing(self):
        manifest = ProviderManifest.from_dict({
            "name": "http_ok",
            "version": "0.1.0",
            "type": "vlm",
            "capabilities": ["vlm.x"],
            "runtime": {"backend": "http", "endpoint": "http://x"},
        })
        provider = GenericProvider(manifest)
        req = ProviderRequest(request_id="r1", capability="vlm.x", inputs={})

        mock_runtime = MagicMock()
        mock_runtime.invoke = AsyncMock(return_value={
            "result": {"boxes": []},
            "confidence": 0.9,
            "status": "done",
            "warnings": ["low_light"],
            "errors": ["minor"],
        })
        provider._runtime = mock_runtime

        resp = await provider.infer(req)
        assert resp.result == {"boxes": []}
        assert resp.confidence == 0.9
        assert resp.status == "done"
        assert resp.warnings == ["low_light"]
        assert resp.errors == ["minor"]

    @pytest.mark.asyncio
    async def test_load_and_unload(self):
        manifest = ProviderManifest.from_dict({
            "name": "py_rt",
            "version": "0.1.0",
            "type": "skill",
            "capabilities": ["skill.x"],
            "runtime": {"backend": "python"},
        })
        provider = GenericProvider(manifest)
        mock_runtime = AsyncMock()
        provider._runtime = mock_runtime

        await provider.load()
        assert provider._healthy is True
        mock_runtime.start.assert_awaited_once()

        await provider.unload()
        assert provider._healthy is False
        mock_runtime.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_health_with_runtime(self):
        manifest = ProviderManifest.from_dict({
            "name": "py_rt",
            "version": "0.1.0",
            "type": "skill",
            "capabilities": ["skill.x"],
            "runtime": {"backend": "python"},
        })
        provider = GenericProvider(manifest)
        mock_runtime = MagicMock()
        mock_runtime._started = True
        provider._runtime = mock_runtime

        h = await provider.health()
        assert h["runtime_started"] is True


# --- ProviderLoader._resolve_provider_class ---

class TestProviderLoaderResolveClass:
    def test_invalid_format_no_dot(self):
        manifest = ProviderManifest.from_dict({
            "name": "bad_fmt",
            "version": "0.1.0",
            "type": "vlm",
            "capabilities": ["vlm.x"],
        })
        manifest.extra = {"provider_class": "NoDot"}
        cls = ProviderLoader._resolve_provider_class(manifest)
        assert cls is GenericProvider

    def test_path_traversal_blocked(self):
        manifest = ProviderManifest.from_dict({
            "name": "traversal",
            "version": "0.1.0",
            "type": "vlm",
            "capabilities": ["vlm.x"],
        })
        manifest.extra = {"provider_class": "os..path.Class"}
        cls = ProviderLoader._resolve_provider_class(manifest)
        assert cls is GenericProvider

    def test_slash_blocked(self):
        manifest = ProviderManifest.from_dict({
            "name": "slash",
            "version": "0.1.0",
            "type": "vlm",
            "capabilities": ["vlm.x"],
        })
        manifest.extra = {"provider_class": "rosclaw/bad.Class"}
        cls = ProviderLoader._resolve_provider_class(manifest)
        assert cls is GenericProvider

    def test_disallowed_prefix(self):
        manifest = ProviderManifest.from_dict({
            "name": "disallowed",
            "version": "0.1.0",
            "type": "vlm",
            "capabilities": ["vlm.x"],
        })
        manifest.extra = {"provider_class": "os.path.Class"}
        cls = ProviderLoader._resolve_provider_class(manifest)
        assert cls is GenericProvider


# --- CapabilityRouter edge cases ---

class TestCapabilityRouterCoverage:
    @pytest.mark.asyncio
    async def test_invoke_primary_fails_fallback_succeeds(self):
        """Primary fails, fallback succeeds — trace marks fallback_used."""
        reg = ProviderRegistry()

        class FailingProvider(Provider):
            name = "primary"
            capabilities = ["llm.chat"]
            async def infer(self, request):  # noqa: E306
                return ProviderResponse(
                    request_id=request.request_id,
                    provider=self.name,
                    capability=request.capability,
                    status="failed",
                    errors=["primary boom"],
                )

        class OKProvider(Provider):
            name = "fallback"
            capabilities = ["llm.chat"]
            async def infer(self, request):  # noqa: E306
                return ProviderResponse(
                    request_id=request.request_id,
                    provider=self.name,
                    capability=request.capability,
                    result={"ok": True},
                )

        m1 = ProviderManifest(name="primary", version="0.1", type="llm", capabilities=["llm.chat"])
        m2 = ProviderManifest(name="fallback", version="0.1", type="llm", capabilities=["llm.chat"])
        p1 = reg.register(m1, lambda m: FailingProvider(m), auto_load=False)
        p2 = reg.register(m2, lambda m: OKProvider(m), auto_load=False)
        p1._healthy = True
        p2._healthy = True
        reg._health["primary"] = {"ok": True}
        reg._health["fallback"] = {"ok": True}

        router = CapabilityRouter(reg)
        req = ProviderRequest(request_id="r1", capability="llm.chat", inputs={})
        resp = await router.invoke(req)
        assert resp.is_ok
        assert resp.trace.get("fallback_used") is True
        assert resp.trace.get("primary_failed") == "primary"

    @pytest.mark.asyncio
    async def test_invoke_all_failed(self):
        """All providers fail — return primary response with fallbacks_exhausted."""
        reg = ProviderRegistry()

        class FailingProvider(Provider):
            name = "primary"
            capabilities = ["llm.chat"]
            async def infer(self, request):  # noqa: E306
                return ProviderResponse(
                    request_id=request.request_id,
                    provider=self.name,
                    capability=request.capability,
                    status="failed",
                    errors=["fail"],
                )

        m1 = ProviderManifest(name="primary", version="0.1", type="llm", capabilities=["llm.chat"])
        p1 = reg.register(m1, lambda m: FailingProvider(m), auto_load=False)
        p1._healthy = True
        reg._health["primary"] = {"ok": True}

        router = CapabilityRouter(reg)
        req = ProviderRequest(request_id="r1", capability="llm.chat", inputs={})
        resp = await router.invoke(req)
        assert not resp.is_ok
        assert resp.trace.get("fallbacks_exhausted") is True

    @pytest.mark.asyncio
    async def test_score_latency_budget_exceeded(self):
        reg = ProviderRegistry()
        m = ProviderManifest(
            name="slow", version="0.1", type="llm",
            capabilities=["llm.chat"],
            runtime=RuntimeSpec(backend="http", endpoint="http://x"),
        )

        class OKProvider(Provider):
            name = "slow"
            capabilities = ["llm.chat"]
            async def infer(self, request):  # noqa: E306
                return ProviderResponse(request_id="r1", provider="slow", capability="llm.chat")

        p = reg.register(m, lambda m: OKProvider(m), auto_load=False)
        p._healthy = True
        reg._health["slow"] = {"ok": True}

        router = CapabilityRouter(reg)
        req = ProviderRequest(
            request_id="r1", capability="llm.chat",
            inputs={}, constraints={"latency_ms": 50},
        )
        with pytest.raises(ProviderNotFoundError, match="No provider passed constraints"):
            await router.route(req)

    @pytest.mark.asyncio
    async def test_score_safety_strict_bonus(self):
        reg = ProviderRegistry()
        m = ProviderManifest(
            name="safe", version="0.1", type="llm",
            capabilities=["llm.chat"],
            safety=SafetySpec(requires_guard=True, executable=False),
        )

        class OKProvider(Provider):
            name = "safe"
            capabilities = ["llm.chat"]
            async def infer(self, request):  # noqa: E306
                return ProviderResponse(request_id="r1", provider="safe", capability="llm.chat")

        p = reg.register(m, lambda m: OKProvider(m), auto_load=False)
        p._healthy = True
        reg._health["safe"] = {"ok": True}

        router = CapabilityRouter(reg)
        req = ProviderRequest(
            request_id="r1", capability="llm.chat",
            inputs={}, constraints={"safety_level": "STRICT"},
        )
        decision = await router.route(req)
        assert decision.selected_provider == "safe"
        assert decision.score > 1.0  # got bonuses

    def test_infer_input_modality_video(self):
        req = ProviderRequest(request_id="r1", capability="c", inputs={"video": {}})
        assert CapabilityRouter._infer_input_modality(req) == "video"

    def test_infer_input_modality_trajectory(self):
        req = ProviderRequest(request_id="r1", capability="c", inputs={"trajectory": {}})
        assert CapabilityRouter._infer_input_modality(req) == "trajectory"

    def test_infer_input_modality_camera_topic(self):
        req = ProviderRequest(request_id="r1", capability="c", inputs={"camera_topic": "/cam"})
        assert CapabilityRouter._infer_input_modality(req) == "image"

    @pytest.mark.asyncio
    async def test_try_infer_exception(self):
        """_try_infer catches exception and returns failed response."""
        reg = ProviderRegistry()

        class BoomProvider(Provider):
            name = "boom"
            capabilities = ["llm.chat"]
            async def infer(self, request):  # noqa: E306
                raise RuntimeError("infer boom")

        m = ProviderManifest(name="boom", version="0.1", type="llm", capabilities=["llm.chat"])
        p = reg.register(m, lambda m: BoomProvider(m), auto_load=False)
        p._healthy = True
        reg._health["boom"] = {"ok": True}

        router = CapabilityRouter(reg)
        req = ProviderRequest(request_id="r1", capability="llm.chat", inputs={})
        resp = await router._try_infer(p, req, None)
        assert resp.status == "failed"
        assert "infer boom" in resp.errors
        assert resp.latency_ms is not None
