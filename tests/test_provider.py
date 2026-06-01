"""Tests for rosclaw.provider core infrastructure."""

from pathlib import Path

import pytest

from rosclaw.provider.core import (
    CapabilityRouter,
    GuardBlockedError,
    Provider,
    ProviderManifest,
    ProviderNotFoundError,
    ProviderRegistry,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailableError,
    RouterDecision,
    RuntimeAdapterError,
)
from rosclaw.provider.core.errors import CapabilityNotSupportedError, ManifestValidationError
from rosclaw.provider.core.manifest import EmbodimentSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manifest() -> ProviderManifest:
    return ProviderManifest(
        name="test_provider",
        version="0.1.0",
        type="llm",
        capabilities=["llm.chat", "llm.plan"],
    )


@pytest.fixture
def robot_manifest() -> ProviderManifest:
    return ProviderManifest(
        name="robot_provider",
        version="0.1.0",
        type="vlm",
        capabilities=["vlm.ground"],
        embodiment=EmbodimentSpec(supported_robots=["ur5e"]),
    )


class DummyProvider(Provider):
    """Concrete provider for testing."""

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        self._ensure_capability_supported(request.capability)
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={"output": "ok"},
        )


# ---------------------------------------------------------------------------
# ProviderManifest
# ---------------------------------------------------------------------------

class TestProviderManifest:
    def test_from_dict_minimal(self):
        m = ProviderManifest.from_dict({
            "name": "p",
            "version": "1.0",
            "type": "llm",
        })
        assert m.name == "p"
        assert m.capabilities == []

    def test_from_dict_missing_required(self):
        with pytest.raises(ManifestValidationError):
            ProviderManifest.from_dict({"name": "p"})

    def test_supports_capability(self, manifest: ProviderManifest):
        assert manifest.supports_capability("llm.chat")
        assert not manifest.supports_capability("vlm.x")

    def test_supports_robot(self, robot_manifest: ProviderManifest):
        assert robot_manifest.supports_robot("ur5e")
        assert not robot_manifest.supports_robot("any")

    def test_supports_robot_universal(self, manifest: ProviderManifest):
        assert manifest.supports_robot("any")  # empty list means universal

    def test_supports_input_modality_empty(self, manifest: ProviderManifest):
        assert manifest.supports_input_modality("text")  # empty modalities = universal

    def test_to_dict(self, manifest: ProviderManifest):
        d = manifest.to_dict()
        assert d["name"] == "test_provider"
        assert "capabilities" in d

    def test_from_yaml_not_found(self, tmp_path: Path):
        with pytest.raises(ManifestValidationError):
            ProviderManifest.from_yaml(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------

class TestProvider:
    def test_init_from_manifest(self, manifest: ProviderManifest):
        p = DummyProvider(manifest)
        assert p.name == "test_provider"
        assert p.capabilities == ["llm.chat", "llm.plan"]
        assert not p._healthy

    def test_ensure_capability_supported_ok(self, manifest: ProviderManifest):
        p = DummyProvider(manifest)
        p._ensure_capability_supported("llm.chat")  # should not raise

    def test_ensure_capability_supported_fail(self, manifest: ProviderManifest):
        p = DummyProvider(manifest)
        with pytest.raises(CapabilityNotSupportedError):
            p._ensure_capability_supported("vlm.x")

    def test_from_manifest_factory(self, manifest: ProviderManifest):
        p = DummyProvider.from_manifest(manifest)
        assert isinstance(p, DummyProvider)

    def test_repr(self, manifest: ProviderManifest):
        p = DummyProvider(manifest)
        assert "DummyProvider" in repr(p)
        assert "test_provider" in repr(p)

    @pytest.mark.asyncio
    async def test_health_default(self, manifest: ProviderManifest):
        p = DummyProvider(manifest)
        h = await p.health()
        assert h["ok"] is False
        assert h["provider"] == "test_provider"

    @pytest.mark.asyncio
    async def test_describe(self, manifest: ProviderManifest):
        p = DummyProvider(manifest)
        d = await p.describe()
        assert "manifest" in d

    @pytest.mark.asyncio
    async def test_infer_success(self, manifest: ProviderManifest):
        p = DummyProvider(manifest)
        req = ProviderRequest(
            request_id="r1", capability="llm.chat", inputs={"text": "hi"}
        )
        resp = await p.infer(req)
        assert resp.is_ok
        assert resp.provider == "test_provider"

    @pytest.mark.asyncio
    async def test_infer_unsupported(self, manifest: ProviderManifest):
        p = DummyProvider(manifest)
        req = ProviderRequest(
            request_id="r1", capability="vlm.x", inputs={}
        )
        with pytest.raises(CapabilityNotSupportedError):
            await p.infer(req)


# ---------------------------------------------------------------------------
# ProviderRegistry
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_register_and_get(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        p = reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        assert reg.get("test_provider") is p
        assert reg.list_providers() == ["test_provider"]

    def test_register_duplicate_raises(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        with pytest.raises(ProviderNotFoundError):
            reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)

    def test_unregister(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        reg.unregister("test_provider")
        assert reg.list_providers() == []

    def test_get_not_found(self):
        reg = ProviderRegistry()
        with pytest.raises(ProviderNotFoundError):
            reg.get("missing")

    def test_find_by_capability(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        results = reg.find_by_capability("llm.chat", healthy_only=False)
        assert len(results) == 1
        assert results[0].name == "test_provider"

    def test_find_by_capability_healthy_only(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        results = reg.find_by_capability("llm.chat", healthy_only=True)
        assert len(results) == 0  # not healthy yet

    def test_find_by_type(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        assert len(reg.find_by_type("llm")) == 1
        assert len(reg.find_by_type("vlm")) == 0

    def test_get_manifest(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        m = reg.get_manifest("test_provider")
        assert m.name == "test_provider"

    @pytest.mark.asyncio
    async def test_check_health(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        h = await reg.check_health("test_provider")
        assert "ok" in h
        assert "timestamp" in h

    @pytest.mark.asyncio
    async def test_check_health_not_registered(self):
        reg = ProviderRegistry()
        h = await reg.check_health("missing")
        assert h["ok"] is False
        assert h["error"] == "not_registered"

    @pytest.mark.asyncio
    async def test_check_all_health(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        results = await reg.check_all_health()
        assert "test_provider" in results

    def test_statistics(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        stats = reg.get_statistics()
        assert stats["total_providers"] == 1
        assert stats["unhealthy_providers"] == 1
        assert stats["by_type"]["llm"] == 1

    def test_is_healthy(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        assert reg.is_healthy("test_provider") is False


# ---------------------------------------------------------------------------
# CapabilityRouter
# ---------------------------------------------------------------------------

class TestCapabilityRouter:
    @pytest.fixture
    def router(self, manifest: ProviderManifest) -> CapabilityRouter:
        reg = ProviderRegistry()
        reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        return CapabilityRouter(reg)

    @pytest.fixture
    def healthy_router(self, manifest: ProviderManifest) -> CapabilityRouter:
        reg = ProviderRegistry()
        p = reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        p._healthy = True
        reg._health["test_provider"] = {"ok": True}
        return CapabilityRouter(reg)

    @pytest.mark.asyncio
    async def test_route_no_capability(self, router: CapabilityRouter):
        req = ProviderRequest(
            request_id="r1", capability="vlm.x", inputs={}
        )
        with pytest.raises(ProviderNotFoundError):
            await router.route(req)

    @pytest.mark.asyncio
    async def test_route_unhealthy(self, router: CapabilityRouter):
        req = ProviderRequest(
            request_id="r1", capability="llm.chat", inputs={}
        )
        with pytest.raises(ProviderUnavailableError):
            await router.route(req)

    @pytest.mark.asyncio
    async def test_route_success(self, healthy_router: CapabilityRouter):
        req = ProviderRequest(
            request_id="r1", capability="llm.chat", inputs={"text": "hi"}
        )
        decision = await healthy_router.route(req)
        assert isinstance(decision, RouterDecision)
        assert decision.selected_provider == "test_provider"
        assert decision.score > 0

    @pytest.mark.asyncio
    async def test_invoke_success(self, healthy_router: CapabilityRouter):
        req = ProviderRequest(
            request_id="r1", capability="llm.chat", inputs={"text": "hi"}
        )
        resp = await healthy_router.invoke(req)
        assert resp.is_ok
        assert resp.provider == "test_provider"

    @pytest.mark.asyncio
    async def test_invoke_fallback(self, manifest: ProviderManifest):
        reg = ProviderRegistry()
        # primary
        p1 = reg.register(manifest, lambda m: DummyProvider(m), auto_load=False)
        p1._healthy = True
        reg._health["test_provider"] = {"ok": True}
        # fallback
        m2 = ProviderManifest(
            name="fallback", version="0.1", type="llm",
            capabilities=["llm.chat"],
        )
        p2 = reg.register(m2, lambda m: DummyProvider(m), auto_load=False)
        p2._healthy = True
        reg._health["fallback"] = {"ok": True}

        router = CapabilityRouter(reg)
        req = ProviderRequest(
            request_id="r1", capability="llm.chat", inputs={"text": "hi"}
        )
        resp = await router.invoke(req)
        assert resp.is_ok

    def test_infer_input_modality(self):
        req_image = ProviderRequest(request_id="r1", capability="c", inputs={"image": {}})
        req_text = ProviderRequest(request_id="r2", capability="c", inputs={"text": {}})
        req_empty = ProviderRequest(request_id="r3", capability="c", inputs={})
        assert CapabilityRouter._infer_input_modality(req_image) == "image"
        assert CapabilityRouter._infer_input_modality(req_text) == "text"
        assert CapabilityRouter._infer_input_modality(req_empty) == ""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TestProviderErrors:
    def test_provider_error_attrs(self):
        e = ProviderNotFoundError("msg", provider="p", request_id="r1")
        assert e.provider == "p"
        assert e.request_id == "r1"
        assert str(e) == "msg"

    def test_guard_blocked_error(self):
        e = GuardBlockedError("blocked", checks=[{"x": 1}], recommended_action="retry")
        assert e.checks == [{"x": 1}]
        assert e.recommended_action == "retry"

    def test_runtime_adapter_error(self):
        e = RuntimeAdapterError("adapter failed")
        assert str(e) == "adapter failed"


# ---------------------------------------------------------------------------
# ProviderRequest
# ---------------------------------------------------------------------------

class TestProviderRequest:
    def test_required_fields(self):
        with pytest.raises(ValueError):
            ProviderRequest(request_id="", capability="c", inputs={})
        with pytest.raises(ValueError):
            ProviderRequest(request_id="r1", capability="", inputs={})

    def test_properties(self):
        req = ProviderRequest(
            request_id="r1",
            capability="c",
            inputs={},
            context={"robot": "ur5e"},
            constraints={"safety_level": "STRICT", "latency_ms": 100},
        )
        assert req.robot_id == "ur5e"
        assert req.safety_level == "STRICT"
        assert req.latency_budget_ms == 100
        assert req.requires_offline is False


# ---------------------------------------------------------------------------
# ProviderResponse
# ---------------------------------------------------------------------------

class TestProviderResponse:
    def test_is_ok(self):
        r = ProviderResponse(request_id="r1", provider="p", capability="c")
        assert r.is_ok
        r.errors = ["fail"]
        assert not r.is_ok

    def test_is_degraded(self):
        r = ProviderResponse(request_id="r1", provider="p", capability="c")
        assert not r.is_degraded
        r.warnings = ["warn"]
        assert r.is_degraded

    def test_to_dict(self):
        r = ProviderResponse(request_id="r1", provider="p", capability="c", result={"x": 1})
        d = r.to_dict()
        assert d["request_id"] == "r1"
        assert d["result"] == {"x": 1}
