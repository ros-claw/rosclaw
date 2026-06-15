"""v1.0 Integration Sprint — End-to-end closed-loop test.

Validates the integration sprint deliverable:
    e-URDF → Provider → Sandbox/Firewall → Runtime → Practice → Memory → HOW
"""

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig


class TestV10IntegrationSprint:
    """Integration sprint acceptance tests."""

    @pytest.fixture
    def rt(self):
        config = RuntimeConfig(
            robot_id="ur5e",
            enable_firewall=True,
            enable_memory=True,
            enable_practice=True,
            enable_knowledge=True,
            enable_how=True,
            enable_provider=True,
            seekdb_backend="memory",
        )
        runtime = Runtime(config)
        runtime.initialize()
        yield runtime
        runtime.stop()

    def test_01_runtime_initializes(self, rt):
        assert rt.state.name == "READY"
        assert rt.e_urdf is not None
        assert rt.provider_registry is not None
        assert rt.capability_router is not None
        assert rt.firewall is not None
        assert rt.memory is not None
        assert rt.practice is not None
        assert rt.how is not None

    def test_02_eurdf_loaded_and_published(self, rt):
        assert rt._robot_profile is not None
        assert rt._robot_profile.robot_id == "universal_robots_ur5e"

    def test_03_robot_capabilities_registered(self, rt):
        registry = rt.provider_registry
        providers = registry.list_providers()
        assert "robot_capabilities" in providers

    def test_04_capability_invoke(self, rt):
        result = rt.capability_invoke("vlm.object_grounding", {"image": "red_cup.jpg"})
        assert result["status"] == "ok"
        assert "result" in result

    def test_05_plan_action(self, rt):
        perception = {"result": {"objects": [{"label": "red_cup", "bbox_2d": [100, 100, 200, 200]}]}}
        plan = rt.plan_action("pick red cup", perception)
        assert plan["status"] == "ok"
        assert "action" in plan

    def test_06_sandbox_check_allow(self, rt):
        action = {"trajectory": [[0.0, -1.57, 1.57, 0.0, 0.0, 0.0]]}
        check = rt.sandbox_check(action)
        assert "decision" in check
        assert check["decision"] in ("ALLOW", "BLOCK")

    def test_07_execute_and_publish_events(self, rt):
        action = {"trajectory": [[0.0, -1.57, 1.57, 0.0, 0.0, 0.0]]}
        result = rt.execute(action)
        assert "status" in result

    def test_08_practice_record(self, rt):
        rt.practice.record(
            event_id="test_evt_001",
            instruction="pick red cup",
            duration_sec=1.5,
        )

    def test_09_memory_write_praxis_event(self, rt):
        record_id = rt.memory.write_praxis_event({
            "event_id": "test_evt_002",
            "event_type": "praxis",
            "instruction": "place red cup",
            "outcome": "success",
            "duration_sec": 2.0,
        })
        assert isinstance(record_id, str)

    def test_10_how_generate_recovery_hint(self, rt):
        hint = rt.how.generate_recovery_hint("collision detected")
        assert hint is not None
        assert "hint" in hint

    def test_11_memory_write_failure_memory(self, rt):
        record_id = rt.memory.write_failure_memory({
            "failure_id": "fail_001",
            "instruction": "pick red cup",
            "failure_type": "collision",
            "reason": "obstacle in path",
            "duration_sec": 1.0,
        })
        assert isinstance(record_id, str)

    def test_12_full_closed_loop(self, rt):
        """Full closed loop as per integration sprint spec."""
        # 1. Invoke capability
        result = rt.capability_invoke("vlm.object_grounding", {"image": "red_cup.jpg"})
        assert result["status"] == "ok"

        # 2. Plan action
        plan = rt.plan_action("pick red cup", result)
        assert plan["status"] == "ok"
        action = plan["action"]

        # 3. Sandbox check
        check = rt.sandbox_check(action)
        assert "decision" in check

        if check["decision"] == "ALLOW":
            # 4. Execute
            exec_result = rt.execute(action)
            assert exec_result["status"] == "ok"
            # 5. Record practice
            rt.practice.record(instruction="pick red cup", duration_sec=1.0)
            # 6. Write memory
            rt.memory.write_praxis_event({
                "event_id": "loop_evt_001",
                "instruction": "pick red cup",
                "outcome": "success",
                "duration_sec": 1.0,
            })
        else:
            # 7. Recovery hint
            hint = rt.how.generate_recovery_hint(check["reason"])
            assert hint is not None
