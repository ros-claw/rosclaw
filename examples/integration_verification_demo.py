"""ROSClaw v1.0 cross-module integration verification script.

Covers: EventBus, Runtime, DataFlywheel, DashboardMetrics, FirewallValidator,
PracticeRecorder, MemoryInterface/SeekDB, HeuristicEngine, CapabilityClient, PID control.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time

os.environ.setdefault("ROSCLAW_ARTIFACTS", "/tmp/rosclaw_integration_artifacts")

import numpy as np

from rosclaw.control import PIDController, PIDGains
from rosclaw.core import Event, EventBus
from rosclaw.dashboard.metrics import DashboardMetrics
from rosclaw.data.flywheel import DataFlywheel, EventType, RobotState
from rosclaw.e_urdf.parser import RobotModel
from rosclaw.firewall.validator import FirewallValidator, ValidationRequest
from rosclaw.how import HeuristicEngine
from rosclaw.memory import MemoryInterface, SeekDBMemoryClient
from rosclaw.practice import PracticeRecorder
from rosclaw.provider.client import CapabilityClient, TaskPlan
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.router import CapabilityRouter


class Counter:
    def __init__(self):
        self.value = 0
        self.events: list[Event] = []


async def test_event_bus_sync_async():
    print("\n[TEST] EventBus sync + async subscription")
    bus = EventBus()
    counter = Counter()

    def on_sync(event: Event):
        counter.value += 1
        counter.events.append(event)

    async def on_async(event: Event):
        counter.value += 1
        counter.events.append(event)

    bus.subscribe("test.topic", on_sync)
    bus.subscribe_async("test.topic", on_async)
    bus.publish(Event(topic="test.topic", payload={"k": "v"}, trace_id="int_test_001"))
    await asyncio.sleep(0.05)
    assert counter.value == 2, f"expected 2, got {counter.value}"
    print("  PASS: sync+async handlers fired")


def test_data_flywheel():
    print("\n[TEST] DataFlywheel capture + event export")
    with tempfile.TemporaryDirectory() as tmpdir:
        fly = DataFlywheel(
            robot_id="test_bot",
            joint_dof=6,
            storage_path=__import__('pathlib').Path(tmpdir),
        )
        for i in range(200):
            state = RobotState(
                timestamp=time.time(),
                joint_positions=np.sin(i * 0.01) * np.ones(6),
                joint_velocities=np.cos(i * 0.01) * np.ones(6) * 0.1,
                joint_torques=np.zeros(6),
            )
            fly.on_control_cycle(state)
            time.sleep(0.001)

        event_id = fly.trigger_event(
            event_type=EventType.SUCCESS,
            metadata={"task": "integration_test", "instruction": "test"},
            pre_duration_sec=0.05,
            post_duration_sec=0.02,
        )
        time.sleep(0.2)  # let background save thread finish
        assert event_id, "event missing id"
        events_with_paths = [e for e in fly._events if e.event_id == event_id and e.data_paths]
        assert events_with_paths, "event missing exported data paths"
        print(f"  PASS: captured {len(events_with_paths[0].data_paths)} data paths for event {event_id}")


def test_dashboard_metrics():
    print("\n[TEST] DashboardMetrics aggregation")
    metrics = DashboardMetrics()
    metrics.record_provider_call("mock_llm", "llm.chat", 45.0, "ok")
    metrics.record_provider_call("mock_llm", "llm.chat", 120.0, "error")
    stats = metrics.get_provider_stats()
    assert stats["total"] == 2
    assert 0.0 < stats["success_rate"] < 1.0
    print(f"  PASS: provider stats {stats}")


def test_firewall_validator():
    print("\n[TEST] FirewallValidator + e-URDF soft limits")
    robot = RobotModel(
        name="test_arm",
        joints={
            "j0": type("J", (), {"limits": {"lower": -1.0, "upper": 1.0, "velocity": 2.0, "effort": 10.0}})(),
            "j1": type("J", (), {"limits": {"lower": -1.0, "upper": 1.0, "velocity": 2.0, "effort": 10.0}})(),
        },
    )
    bus = EventBus()
    validator = FirewallValidator(robot_model=robot, event_bus=bus, safety_level="STRICT")
    validator.initialize()

    req = ValidationRequest(
        request_id="req_1",
        robot_id="test_arm",
        trajectory=[[0.0, 0.0], [0.5, 0.5], [2.0, 0.0]],
    )
    resp = validator.validate(req)
    assert not resp.is_safe, "expected unsafe trajectory to be rejected"
    assert any(v.joint_index == 0 for v in resp.violations)
    print(f"  PASS: firewall rejected unsafe trajectory with {resp.violation_count} violations")
    validator.stop()


def test_practice_memory():
    print("\n[TEST] PracticeRecorder + MemoryInterface + SeekDB")
    bus = EventBus()
    _recorder = PracticeRecorder(robot_id="test_bot", event_bus=bus)
    memory = MemoryInterface(robot_id="test_bot", event_bus=bus)
    memory.initialize()

    bus.publish(Event(topic="rosclaw.practice.event.created", payload={"tag": "grasp_attempt", "event_id": "p001"}))
    memory.store_experience(
        event_id="exp_001",
        event_type="skill_execution",
        instruction="grasp the red cup",
        outcome="success",
        tags=["grasp", "cup"],
        metadata={"reward": 0.95},
    )
    # allow preloader or direct query
    results = memory.find_similar_experiences("grasp cup", limit=5)
    print(f"  PASS: stored and retrieved {len(results)} memory entries")
    memory.stop()


async def test_how_recovery():
    print("\n[TEST] HeuristicEngine failure recovery hint")
    client = SeekDBMemoryClient()
    how = HeuristicEngine(seekdb_client=client)
    await how.seed_defaults()
    hint = await how.suggest_recovery(
        error_log="joint limit exceeded during reach",
        context={"robot_id": "ur5e", "task": "pick"},
    )
    assert hint
    print(f"  PASS: recovery hint returned action={hint.get('action', 'n/a')}")


async def test_capability_client():
    print("\n[TEST] CapabilityClient composite task routing")
    registry = ProviderRegistry()
    router = CapabilityRouter(registry)

    from rosclaw.provider.core.provider import Provider
    from rosclaw.provider.core.request import ProviderRequest
    from rosclaw.provider.core.response import ProviderResponse

    class MockProvider(Provider):
        name = "mock_test"
        version = "1.0"
        capabilities = ["skill.mock_grasp"]

        def __init__(self, manifest):
            super().__init__(manifest)
            self._healthy = True

        async def infer(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(
                request_id=request.request_id,
                provider=self.name,
                capability=request.capability,
                result={"success": True},
                status="ok",
            )

    manifest = ProviderManifest(
        name="mock_test",
        type="skill",
        version="1.0",
        capabilities=["skill.mock_grasp"],
    )
    registry.register(manifest, lambda m: MockProvider(m))
    client = CapabilityClient(router)

    def planner(task, robot, scene_input, safety_level):
        return TaskPlan(
            task=task,
            steps=[{"capability": "skill.mock_grasp", "inputs": {}, "critical": True}],
        )

    client._plan_task = planner
    result = await client.run_task(task="grasp the cup", robot="ur5e")

    assert result.status == "success"
    print(f"  PASS: composite task status={result.status}")


def test_pid_control():
    print("\n[TEST] PIDController convergence")
    pid = PIDController(PIDGains(kp=8.0, ki=0.5, kd=0.2))
    target = 1.0
    state = 0.0
    for _ in range(200):
        cmd = pid.update(target - state, dt=0.01)
        state += cmd * 0.01
    error = abs(target - state)
    assert error < 0.05, f"PID did not converge: error={error}"
    print(f"  PASS: PID converged to error={error:.4f}")


async def main():
    print("=" * 60)
    print("ROSClaw v1.0 Cross-Module Integration Verification")
    print("=" * 60)

    await test_event_bus_sync_async()
    test_data_flywheel()
    test_dashboard_metrics()
    test_firewall_validator()
    test_practice_memory()
    await test_how_recovery()
    await test_capability_client()
    test_pid_control()

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
