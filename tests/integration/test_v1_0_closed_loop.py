"""ROSClaw v1.0 E2E Closed Loop Integration Test.

Full pipeline:
    e-URDF load → Provider call → Sandbox check → Runtime execute
    → Practice record → Memory write → HOW recovery
"""

import asyncio
import pytest

from rosclaw.core.event_bus import EventBus, Event
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.core.router import CapabilityRouter
from rosclaw.runtime import RobotRegistry
from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter
from rosclaw.memory import MemoryInterface
from rosclaw.how import HeuristicEngine


class MockVLMProvider(Provider):
    name = "qwen_vl"
    version = "0.1.0"
    capabilities = ["vlm.object_grounding"]

    async def infer(self, request):
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={"objects": [{"label": "red_cup", "bbox": [0.1, 0.2, 0.3, 0.4]}]},
            status="ok",
        )


class MockSkillProvider(Provider):
    name = "moveit_skill"
    version = "0.1.0"
    capabilities = ["skill.pick_and_place"]

    async def infer(self, request):
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={
                "trajectory": [
                    [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
                    [0.1, -1.4, 1.4, 0.1, 0.0, 0.0],
                ],
                "gripper": "close",
            },
            status="ok",
        )


class MockCriticProvider(Provider):
    name = "critic_vlm"
    version = "0.1.0"
    capabilities = ["critic.success_detection"]

    async def infer(self, request):
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result={"success": True, "confidence": 0.92},
            status="ok",
        )


class TestV1_0ClosedLoop:
    """End-to-end closed loop: e-URDF → Provider → Sandbox → Memory → HOW."""

    @pytest.fixture
    async def e2e_setup(self):
        # 1. e-URDF
        robot_reg = RobotRegistry()
        profile = robot_reg.install("ur5e")

        # 2. EventBus
        event_bus = EventBus()

        # 3. Providers
        provider_reg = ProviderRegistry(event_bus=event_bus)
        vlm_manifest = ProviderManifest.from_dict({
            "name": "qwen_vl",
            "version": "0.1.0",
            "type": "vlm",
            "capabilities": ["vlm.object_grounding"],
            "embodiment": {"supported_robots": [profile.robot_id]},
            "safety": {"executable": False, "requires_guard": True},
        })
        skill_manifest = ProviderManifest.from_dict({
            "name": "moveit_skill",
            "version": "0.1.0",
            "type": "skill",
            "capabilities": ["skill.pick_and_place"],
            "embodiment": {"supported_robots": [profile.robot_id]},
            "safety": {"executable": True, "requires_guard": True},
        })
        critic_manifest = ProviderManifest.from_dict({
            "name": "critic_vlm",
            "version": "0.1.0",
            "type": "critic",
            "capabilities": ["critic.success_detection"],
            "embodiment": {"supported_robots": [profile.robot_id]},
            "safety": {"executable": False, "requires_guard": False},
        })

        provider_reg._providers["qwen_vl"] = MockVLMProvider(vlm_manifest)
        provider_reg._health["qwen_vl"] = {"ok": True}
        provider_reg._manifests["qwen_vl"] = vlm_manifest

        provider_reg._providers["moveit_skill"] = MockSkillProvider(skill_manifest)
        provider_reg._health["moveit_skill"] = {"ok": True}
        provider_reg._manifests["moveit_skill"] = skill_manifest

        provider_reg._providers["critic_vlm"] = MockCriticProvider(critic_manifest)
        provider_reg._health["critic_vlm"] = {"ok": True}
        provider_reg._manifests["critic_vlm"] = critic_manifest

        router = CapabilityRouter(provider_reg)

        # 4. Sandbox
        sandbox = SandboxRuntimeAdapter(
            config={"engine": "mujoco", "world_id": "tabletop", "robot_id": "universal_robots_ur5e"},
            event_bus=event_bus,
            e_urdf_model=profile.embodiment,
        )
        sandbox.initialize()

        # 5. Memory
        memory = MemoryInterface(robot_id=profile.robot_id)

        # 6. HOW — seed defaults for rule-based recovery
        how = HeuristicEngine(seekdb_client=memory.seekdb_client)
        await how.initialize()
        await how.seed_defaults()

        yield {
            "profile": profile,
            "event_bus": event_bus,
            "provider_reg": provider_reg,
            "router": router,
            "sandbox": sandbox,
            "memory": memory,
            "how": how,
        }

        sandbox.stop()

    @pytest.mark.asyncio
    async def test_01_eurdf_loaded(self, e2e_setup):
        ctx = e2e_setup
        profile = ctx["profile"]
        assert profile.robot_id == "universal_robots_ur5e"
        assert profile.embodiment.dof == 6
        assert len(profile.capability.capabilities) >= 3
        assert profile.safety.safety_level == "STRICT"

    @pytest.mark.asyncio
    async def test_02_provider_routes_with_robot_context(self, e2e_setup):
        ctx = e2e_setup
        profile = ctx["profile"]
        router = ctx["router"]
        request = ProviderRequest(
            request_id="e2e_001",
            capability="vlm.object_grounding",
            inputs={"image": "base64_red_cup_scene"},
            context={"robot": profile.robot_id, "task_id": "pick_red_cup"},
            constraints={"safety_level": "STRICT"},
        )
        decision = await router.route(request)
        assert decision.selected_provider == "qwen_vl"

    @pytest.mark.asyncio
    async def test_03_provider_invokes_and_returns_result(self, e2e_setup):
        ctx = e2e_setup
        profile = ctx["profile"]
        router = ctx["router"]
        request = ProviderRequest(
            request_id="e2e_002",
            capability="vlm.object_grounding",
            inputs={"image": "base64..."},
            context={"robot": profile.robot_id},
        )
        response = await router.invoke(request)
        assert response.is_ok
        assert response.provider == "qwen_vl"
        assert "objects" in response.result
        assert response.result["objects"][0]["label"] == "red_cup"

    @pytest.mark.asyncio
    async def test_04_skill_provider_generates_trajectory(self, e2e_setup):
        ctx = e2e_setup
        profile = ctx["profile"]
        router = ctx["router"]
        request = ProviderRequest(
            request_id="e2e_003",
            capability="skill.pick_and_place",
            inputs={"object": "red_cup", "location": "table_center"},
            context={"robot": profile.robot_id},
        )
        response = await router.invoke(request)
        assert response.is_ok
        assert "trajectory" in response.result
        assert len(response.result["trajectory"]) > 0

    @pytest.mark.asyncio
    async def test_05_sandbox_validates_trajectory(self, e2e_setup):
        ctx = e2e_setup
        sandbox = ctx["sandbox"]
        trajectory = [[0.0, -1.57, 1.57, 0.0, 0.0, 0.0]]
        result = sandbox.validate_trajectory(trajectory, safety_level="STRICT")
        assert isinstance(result, dict)
        assert "is_safe" in result

    @pytest.mark.asyncio
    async def test_06_memory_records_event(self, e2e_setup):
        ctx = e2e_setup
        memory = ctx["memory"]
        event = {
            "type": "task_execution",
            "robot_id": "universal_robots_ur5e",
            "task_id": "pick_red_cup",
            "provider": "moveit_skill",
            "status": "success",
        }
        memory.seekdb_client.insert("events", event)
        results = memory.seekdb_client.query("events", filters={"task_id": "pick_red_cup"})
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_07_how_suggests_recovery(self, e2e_setup):
        ctx = e2e_setup
        how = ctx["how"]
        suggestion = await how.suggest_recovery("joint_limit_exceeded")
        assert suggestion is not None
        assert "action" in suggestion

    @pytest.mark.asyncio
    async def test_08_critic_evaluates_success(self, e2e_setup):
        ctx = e2e_setup
        profile = ctx["profile"]
        router = ctx["router"]
        request = ProviderRequest(
            request_id="e2e_004",
            capability="critic.success_detection",
            inputs={"scene_image": "base64...", "expected_object": "red_cup_at_target"},
            context={"robot": profile.robot_id, "task_id": "pick_red_cup"},
        )
        response = await router.invoke(request)
        assert response.is_ok
        assert response.result.get("success") is True
        assert response.result.get("confidence", 0) > 0.8

    @pytest.mark.asyncio
    async def test_09_event_bus_publishes(self, e2e_setup):
        ctx = e2e_setup
        event_bus = ctx["event_bus"]
        received = []
        def subscriber(event):  # noqa: E306
            received.append(event)
        event_bus.subscribe("test.topic", subscriber)
        event_bus.publish(Event("test.topic", {"msg": "hello"}, source="test"))
        await asyncio.sleep(0.01)
        assert len(received) == 1
        assert received[0].payload["msg"] == "hello"

    @pytest.mark.asyncio
    async def test_10_full_closed_loop(self, e2e_setup):
        ctx = e2e_setup
        profile = ctx["profile"]
        router = ctx["router"]
        sandbox = ctx["sandbox"]
        memory = ctx["memory"]
        how = ctx["how"]

        # Perception
        vlm_request = ProviderRequest(
            request_id="closed_loop_001",
            capability="vlm.object_grounding",
            inputs={"image": "scene_camera_001"},
            context={"robot": profile.robot_id, "task_id": "pick_red_cup"},
            constraints={"safety_level": "STRICT"},
        )
        vlm_resp = await router.invoke(vlm_request)
        assert vlm_resp.is_ok
        detected_objects = vlm_resp.result.get("objects", [])
        assert any(o["label"] == "red_cup" for o in detected_objects)

        # Skill Generation
        skill_request = ProviderRequest(
            request_id="closed_loop_002",
            capability="skill.pick_and_place",
            inputs={"object": "red_cup", "location": "table_center"},
            context={"robot": profile.robot_id, "task_id": "pick_red_cup"},
        )
        skill_resp = await router.invoke(skill_request)
        assert skill_resp.is_ok
        trajectory = skill_resp.result.get("trajectory", [])

        # Sandbox Validation
        validation = sandbox.validate_trajectory(trajectory, safety_level="STRICT")
        assert "is_safe" in validation

        # Memory Record
        memory.seekdb_client.insert("episodes", {
            "episode_id": "ep_001",
            "robot_id": profile.robot_id,
            "task_id": "pick_red_cup",
            "trajectory": trajectory,
            "sandbox_validation": validation,
            "vlm_result": vlm_resp.result,
        })

        # Critic Evaluation
        critic_request = ProviderRequest(
            request_id="closed_loop_003",
            capability="critic.success_detection",
            inputs={"scene_after": "scene_camera_002"},
            context={"robot": profile.robot_id, "task_id": "pick_red_cup"},
        )
        critic_resp = await router.invoke(critic_request)
        assert critic_resp.is_ok

        # HOW Recovery
        if not critic_resp.result.get("success", False):
            recovery = await how.suggest_recovery("grasp_slippage")
            assert recovery is not None
        else:
            recovery = await how.suggest_recovery("joint_limit_exceeded")
            assert recovery is not None

        # Verify Memory
        episodes = memory.seekdb_client.query("episodes", filters={"episode_id": "ep_001"})
        assert len(episodes) >= 1
        assert episodes[0]["robot_id"] == profile.robot_id

        print("\n✅ v1.0 Closed Loop: e-URDF → Provider → Sandbox → Memory → HOW — ALL PASS")
