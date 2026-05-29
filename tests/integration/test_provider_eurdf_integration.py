"""Provider + e-URDF Integration Tests.

Tests that ProviderRegistry and CapabilityRouter can:
1. Read RobotCapabilityProfile from e-URDF
2. Route requests based on robot capabilities
3. Reject unsafe requests based on robot safety_limits
"""

import pytest

from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.core.router import CapabilityRouter
from rosclaw.provider.core.errors import ProviderNotFoundError
from rosclaw.runtime import RobotRegistry


class MockVLMProvider(Provider):
    name = "mock_vlm"
    version = "0.1.0"
    capabilities = ["vlm.object_grounding", "vlm.scene_understanding"]

    async def infer(self, request):
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={"objects": [{"label": "red_cup", "bbox": [0.1, 0.2, 0.3, 0.4]}]},
            status="ok",
        )


class MockSkillProvider(Provider):
    name = "mock_skill"
    version = "0.1.0"
    capabilities = ["skill.pick_and_place", "skill.push"]

    async def infer(self, request):
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={"trajectory": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]]},
            status="ok",
        )


class MockUnsafeProvider(Provider):
    name = "mock_unsafe"
    version = "0.1.0"
    capabilities = ["skill.pick_and_place"]

    async def infer(self, request):
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={"action": "move_directly"},
            status="ok",
        )


@pytest.fixture
def robot_registry():
    return RobotRegistry()


@pytest.fixture
def provider_registry():
    return ProviderRegistry()


@pytest.fixture
def mock_providers():
    vlm_manifest = ProviderManifest.from_dict({
        "name": "mock_vlm",
        "version": "0.1.0",
        "type": "vlm",
        "capabilities": ["vlm.object_grounding", "vlm.scene_understanding"],
        "embodiment": {"supported_robots": ["universal_robots_ur5e", "unitree_go2"]},
        "safety": {"executable": False, "requires_guard": True},
    })
    skill_manifest = ProviderManifest.from_dict({
        "name": "mock_skill",
        "version": "0.1.0",
        "type": "skill",
        "capabilities": ["skill.pick_and_place", "skill.push"],
        "embodiment": {"supported_robots": ["universal_robots_ur5e"]},
        "safety": {"executable": True, "requires_guard": True},
    })
    unsafe_manifest = ProviderManifest.from_dict({
        "name": "mock_unsafe",
        "version": "0.1.0",
        "type": "skill",
        "capabilities": ["skill.pick_and_place"],
        "embodiment": {"supported_robots": []},
        "safety": {"executable": True, "requires_guard": False},
    })
    return {
        "vlm": MockVLMProvider(vlm_manifest),
        "skill": MockSkillProvider(skill_manifest),
        "unsafe": MockUnsafeProvider(unsafe_manifest),
        "manifests": {"vlm": vlm_manifest, "skill": skill_manifest, "unsafe": unsafe_manifest},
    }


class TestProviderEURDFIntegration:
    @pytest.mark.asyncio
    async def test_provider_registry_reads_robot_capabilities(self, robot_registry, provider_registry, mock_providers):
        profile = robot_registry.install("ur5e")
        cap_names = [c["name"] for c in profile.capability.capabilities]
        reg = provider_registry
        reg._providers["mock_vlm"] = mock_providers["vlm"]
        reg._health["mock_vlm"] = {"ok": True}
        reg._providers["mock_skill"] = mock_providers["skill"]
        reg._health["mock_skill"] = {"ok": True}
        assert "pick_and_place" in cap_names
        assert "push" in cap_names
        vlm = reg.get("mock_vlm")
        assert vlm is not None
        assert "vlm.object_grounding" in vlm.capabilities

    @pytest.mark.asyncio
    async def test_capability_router_matches_robot(self, robot_registry, mock_providers):
        profile = robot_registry.install("ur5e")
        robot_id = profile.robot_id
        reg = ProviderRegistry()
        reg._providers["mock_vlm"] = mock_providers["vlm"]
        reg._health["mock_vlm"] = {"ok": True}
        reg._manifests["mock_vlm"] = mock_providers["manifests"]["vlm"]
        reg._providers["mock_skill"] = mock_providers["skill"]
        reg._health["mock_skill"] = {"ok": True}
        reg._manifests["mock_skill"] = mock_providers["manifests"]["skill"]
        router = CapabilityRouter(reg)
        request = ProviderRequest(
            request_id="req_001",
            capability="vlm.object_grounding",
            inputs={"image": "base64..."},
            context={"robot": robot_id, "task_id": "pick_red_cup"},
        )
        decision = await router.route(request)
        assert decision.selected_provider == "mock_vlm"

    @pytest.mark.asyncio
    async def test_capability_router_rejects_unsupported_robot(self, robot_registry, mock_providers):
        reg = ProviderRegistry()
        reg._providers["mock_skill"] = mock_providers["skill"]
        reg._health["mock_skill"] = {"ok": True}
        reg._manifests["mock_skill"] = mock_providers["manifests"]["skill"]
        router = CapabilityRouter(reg)
        request = ProviderRequest(
            request_id="req_002",
            capability="skill.pick_and_place",
            inputs={"object": "red_cup"},
            context={"robot": "unsupported_robot_xyz"},
        )
        with pytest.raises(ProviderNotFoundError):
            await router.route(request)

    @pytest.mark.asyncio
    async def test_safety_level_strict_prefers_guarded_providers(self, robot_registry, mock_providers):
        reg = ProviderRegistry()
        reg._providers["mock_skill"] = mock_providers["skill"]
        reg._health["mock_skill"] = {"ok": True}
        reg._manifests["mock_skill"] = mock_providers["manifests"]["skill"]
        reg._providers["mock_unsafe"] = mock_providers["unsafe"]
        reg._health["mock_unsafe"] = {"ok": True}
        reg._manifests["mock_unsafe"] = mock_providers["manifests"]["unsafe"]
        router = CapabilityRouter(reg)
        profile = robot_registry.install("ur5e")
        request = ProviderRequest(
            request_id="req_003",
            capability="skill.pick_and_place",
            inputs={"object": "red_cup"},
            context={"robot": profile.robot_id},
            constraints={"safety_level": "STRICT"},
        )
        decision = await router.route(request)
        assert decision.selected_provider == "mock_skill"

    @pytest.mark.asyncio
    async def test_provider_invokes_with_robot_context(self, robot_registry, mock_providers):
        reg = ProviderRegistry()
        reg._providers["mock_vlm"] = mock_providers["vlm"]
        reg._health["mock_vlm"] = {"ok": True}
        reg._manifests["mock_vlm"] = mock_providers["manifests"]["vlm"]
        router = CapabilityRouter(reg)
        profile = robot_registry.install("ur5e")
        request = ProviderRequest(
            request_id="req_004",
            capability="vlm.object_grounding",
            inputs={"image": "base64..."},
            context={"robot": profile.robot_id, "task_id": "pick_red_cup", "workspace": "tabletop"},
        )
        response = await router.invoke(request)
        assert response.is_ok
        assert response.provider == "mock_vlm"
        assert response.capability == "vlm.object_grounding"
        assert "objects" in response.result


class TestProviderSafetyIntegration:
    def test_robot_safety_limits_loaded(self, robot_registry):
        profile = robot_registry.install("ur5e")
        safety = profile.safety
        assert safety.safety_level == "STRICT"
        assert safety.pfl.get("max_tcp_force") == 150.0
        assert safety.pfl.get("max_tcp_torque") == 8.0
        assert safety.collision_detection.get("threshold_force") == 50.0

    def test_robot_workspace_limits(self, robot_registry):
        profile = robot_registry.install("ur5e")
        workspace = profile.safety.workspace_boundaries
        assert workspace.get("type") == "enclosure"
        assert workspace.get("fenceless") is True

    def test_capability_constraints_match_safety(self, robot_registry):
        profile = robot_registry.install("ur5e")
        caps = profile.capability.capabilities
        pick = next(c for c in caps if c["name"] == "pick_and_place")
        constraints = pick.get("constraints", {})
        assert constraints.get("max_payload") == 5.0
        assert constraints.get("max_reach") == 0.85

    @pytest.mark.asyncio
    async def test_provider_rejects_exceeding_force_limit(self, robot_registry, mock_providers):
        profile = robot_registry.install("ur5e")
        max_force = profile.safety.pfl.get("max_tcp_force", 150.0)

        class ForceLimitGuardProvider(Provider):
            name = "force_guard"
            version = "0.1.0"
            capabilities = ["skill.force_compliant_insert"]

            async def infer(self, request):
                requested_force = request.inputs.get("force", 0)
                if requested_force > max_force:
                    return ProviderResponse(
                        request_id=request.request_id,
                        provider=self.name,
                        capability=request.capability,
                        status="blocked",
                        errors=[f"Requested force {requested_force}N exceeds robot limit {max_force}N"],
                    )
                return ProviderResponse(
                    request_id=request.request_id,
                    provider=self.name,
                    capability=request.capability,
                    result={"force": requested_force},
                    status="ok",
                )

        manifest = ProviderManifest.from_dict({
            "name": "force_guard",
            "version": "0.1.0",
            "type": "skill",
            "capabilities": ["skill.force_compliant_insert"],
        })
        provider = ForceLimitGuardProvider(manifest)
        safe_request = ProviderRequest(
            request_id="req_safe",
            capability="skill.force_compliant_insert",
            inputs={"force": 100.0},
            context={"robot": profile.robot_id},
        )
        safe_resp = await provider.infer(safe_request)
        assert safe_resp.is_ok
        unsafe_request = ProviderRequest(
            request_id="req_unsafe",
            capability="skill.force_compliant_insert",
            inputs={"force": 200.0},
            context={"robot": profile.robot_id},
        )
        unsafe_resp = await provider.infer(unsafe_request)
        assert not unsafe_resp.is_ok
        assert "exceeds robot limit" in unsafe_resp.errors[0]
