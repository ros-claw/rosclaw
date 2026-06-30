"""Tests for the PhysicalReasoner abstraction."""

from __future__ import annotations

import pytest

from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.reasoner import (
    CosmosReasoner,
    GeminiReasoner,
    QwenReasoner,
    get_reasoner,
)
from rosclaw.provider.reasoners.base import PhysicalReasoner


class TestPhysicalReasonerFactory:
    """Factory resolution."""

    def test_default_reasoner_is_cosmos(self) -> None:
        reasoner = get_reasoner()
        assert isinstance(reasoner, CosmosReasoner)
        assert reasoner.name == "cosmos"

    def test_cosmos_by_id(self) -> None:
        for pid in ("cosmos", "gpu_cosmos", "COSMOS", "Cosmos"):
            reasoner = get_reasoner(pid)
            assert isinstance(reasoner, CosmosReasoner)
        assert get_reasoner("gpu_cosmos").name == "cosmos"

    def test_gemini_stub_by_id(self) -> None:
        reasoner = get_reasoner("gemini")
        assert isinstance(reasoner, GeminiReasoner)
        assert reasoner.name == "gemini"

    def test_qwen_stub_by_id(self) -> None:
        reasoner = get_reasoner("qwen")
        assert isinstance(reasoner, QwenReasoner)
        assert reasoner.name == "qwen"

    def test_unknown_id_falls_back_to_cosmos(self) -> None:
        reasoner = get_reasoner("unknown_provider")
        assert isinstance(reasoner, CosmosReasoner)
        assert reasoner.name == "unknown_provider"


class TestReasonerInterface:
    """Every PhysicalReasoner exposes reason/plan/analyze."""

    def test_cosmos_interface(self) -> None:
        reasoner = CosmosReasoner(endpoint="http://localhost:9")
        assert isinstance(reasoner, PhysicalReasoner)

        response = reasoner.reason("What is this?")
        assert response.provider == "cosmos"
        assert response.status == "failed"
        assert any("unreachable" in e.lower() or "connection" in e.lower() for e in response.errors)

        plan_resp = reasoner.plan("pick the red cube")
        assert plan_resp.provider == "cosmos"

        analyze_resp = reasoner.analyze([{"object": "cube"}])
        assert analyze_resp.provider == "cosmos"

    def test_gemini_stub_returns_failed(self) -> None:
        reasoner = get_reasoner("gemini")
        for method in (reasoner.reason, reasoner.plan, reasoner.analyze):
            response = method("input")
            assert response.status == "failed"
            assert "not yet implemented" in response.errors[0]

    def test_qwen_stub_returns_failed(self) -> None:
        reasoner = get_reasoner("qwen")
        response = reasoner.reason("What is this?")
        assert response.status == "failed"
        assert "not yet implemented" in response.errors[0]


class TestRegistryReasoner:
    """ProviderRegistry.get_reasoner bridges registry and factory."""

    def test_unregistered_provider_uses_factory(self) -> None:
        registry = ProviderRegistry()
        reasoner = registry.get_reasoner("gemini")
        assert isinstance(reasoner, GeminiReasoner)

    def test_registered_provider_uses_manifest_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types

        from rosclaw.provider.core.manifest import ProviderManifest, RuntimeSpec

        registry = ProviderRegistry()
        manifest = ProviderManifest(
            name="local_cosmos",
            version="0.0.1",
            type="cosmos",
            capabilities=["vlm.risk_assessment"],
            runtime=RuntimeSpec(endpoint="http://localhost:9"),
        )

        def factory(m: ProviderManifest) -> object:
            return types.SimpleNamespace(_healthy=True)

        registry.register(manifest, factory, auto_load=False)
        reasoner = registry.get_reasoner("local_cosmos")
        assert isinstance(reasoner, CosmosReasoner)
