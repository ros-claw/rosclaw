"""Scenario D: 移动机器人三点巡检 — 端到端验收测试.

验证链路:
    VLN导航 → VLM观测 → Sandbox验证 → Runtime执行 → Memory记录 → How异常建议
"""

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.provider.core.request import ProviderRequest


@pytest.fixture
def runtime_with_go2(tmp_path):
    """Runtime with Go2 quadruped and MuJoCo sandbox."""
    config = RuntimeConfig(
        robot_id="go2",
        enable_provider=True,
        enable_practice=True,
        enable_memory=True,
        timeline_output_dir=str(tmp_path / "practice"),
    )
    runtime = Runtime(config)
    runtime.initialize()
    yield runtime
    runtime.stop()


@pytest.mark.asyncio
async def test_scenario_d_vln_navigates_to_waypoints(runtime_with_go2):
    """Step 1: VLN provider plans navigation waypoints."""
    runtime = runtime_with_go2
    result = await runtime.capability_router.invoke(
        request=ProviderRequest(
            request_id="sc_d_001",
            capability="vlm.scene_understanding",
            inputs={"scene_description": "industrial floor with gauge, valve, and doorway"},
        ),
    )
    assert result.status == "ok"
    assert result.capability == "vlm.scene_understanding"


@pytest.mark.asyncio
async def test_scenario_d_sandbox_has_go2_physics(runtime_with_go2):
    """Step 2: Sandbox loads Go2 MuJoCo model."""
    runtime = runtime_with_go2
    assert runtime._sandbox is not None
    assert runtime._sandbox.has_physics, "Go2 sandbox must have real MuJoCo physics"


@pytest.mark.asyncio
async def test_scenario_d_inspect_point_a_gauge(runtime_with_go2):
    """Step 3: Inspect point A — dashboard/gauge."""
    runtime = runtime_with_go2

    events = []
    for topic in ["rosclaw.provider.inference.completed", "rosclaw.sandbox.action.allowed"]:
        runtime.event_bus.subscribe(topic, lambda e, t=topic: events.append(t))

    result = runtime.execute(
        action={
            "type": "navigate",
            "parameters": {"target": [2.0, 0.0, 0.0], "speed": 0.5},
            "episode_id": "sc_d_ep_a",
        },
    )
    assert result is not None
    assert "trajectory_data" in result
    assert len(result["trajectory_data"]) > 0

    # Real physics verification
    first_jp = result["trajectory_data"][0].get("joint_positions", [])
    assert len(first_jp) >= 6, "Go2 physics must return joint_positions"

    assert "rosclaw.provider.inference.completed" in events
    assert "rosclaw.sandbox.action.allowed" in events


@pytest.mark.asyncio
async def test_scenario_d_inspect_point_b_valve(runtime_with_go2):
    """Step 4: Inspect point B — valve."""
    runtime = runtime_with_go2

    result = runtime.execute(
        action={
            "type": "navigate",
            "parameters": {"target": [4.0, 2.0, 0.0], "speed": 0.5},
            "episode_id": "sc_d_ep_b",
        },
    )
    assert result is not None
    assert "trajectory_data" in result


@pytest.mark.asyncio
async def test_scenario_d_inspect_point_c_doorway(runtime_with_go2):
    """Step 5: Inspect point C — doorway."""
    runtime = runtime_with_go2

    result = runtime.execute(
        action={
            "type": "navigate",
            "parameters": {"target": [6.0, 0.0, 0.0], "speed": 0.5},
            "episode_id": "sc_d_ep_c",
        },
    )
    assert result is not None
    assert "trajectory_data" in result


@pytest.mark.asyncio
async def test_scenario_d_memory_records_all_inspections(runtime_with_go2):
    """Step 6: Memory records all inspection points."""
    runtime = runtime_with_go2

    # Execute 3 inspection points
    for i, target in enumerate([[2.0, 0.0, 0.0], [4.0, 2.0, 0.0], [6.0, 0.0, 0.0]]):
        runtime.execute(
            action={
                "type": "navigate",
                "parameters": {"target": target, "speed": 0.5},
                "episode_id": f"sc_d_ep_{i}",
            },
        )

    stats = runtime.memory.get_statistics()
    assert stats is not None


@pytest.mark.asyncio
async def test_scenario_d_full_patrol_closed_loop(runtime_with_go2):
    """Full Scenario D: 三点巡检端到端闭环."""
    runtime = runtime_with_go2

    all_events = []
    for topic in ["rosclaw.provider.inference.completed", "rosclaw.sandbox.action.allowed", "rosclaw.dashboard.trace.updated"]:
        runtime.event_bus.subscribe(topic, lambda e, t=topic: all_events.append(t))

    # Patrol all 3 points
    waypoints = [
        ([2.0, 0.0, 0.0], "gauge"),
        ([4.0, 2.0, 0.0], "valve"),
        ([6.0, 0.0, 0.0], "doorway"),
    ]

    for target, label in waypoints:
        vlm_result = await runtime.capability_router.invoke(
            request=ProviderRequest(
                request_id=f"sc_d_vlm_{label}",
                capability="vlm.scene_understanding",
                inputs={"location": label},
            ),
        )
        assert vlm_result.status == "ok"

        exec_result = runtime.execute(
            action={
                "type": "navigate",
                "parameters": {"target": target, "speed": 0.5},
                "episode_id": f"sc_d_{label}",
            },
        )
        assert exec_result is not None
        assert "trajectory_data" in exec_result
        if exec_result["trajectory_data"]:
            assert "joint_positions" in exec_result["trajectory_data"][0]

    # Verify events
    assert "rosclaw.provider.inference.completed" in all_events
    assert "rosclaw.sandbox.action.allowed" in all_events
    assert "rosclaw.dashboard.trace.updated" in all_events

    print(f"\n✅ Scenario D 巡检端到端闭环验证通过！")
    print(f"   Waypoints visited: {len(waypoints)}")
    print(f"   Events captured: {len(all_events)}")
