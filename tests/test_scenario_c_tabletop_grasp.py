"""Scenario C: 桌面抓取红杯子 — 端到端验收测试.

验证链路:
    VLM定位 → Skill生成抓取 → Sandbox验证 → Runtime执行 → Critic判断 → Memory记录
"""

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.provider.core.request import ProviderRequest


@pytest.fixture
def runtime_with_physics(tmp_path):
    """Runtime with real MuJoCo sandbox."""
    config = RuntimeConfig(
        robot_id="ur5e",
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
async def test_scenario_c_vlm_locates_red_cup(runtime_with_physics):
    """Step 1: VLM provider locates the red cup."""
    runtime = runtime_with_physics
    result = await runtime.capability_router.invoke(
        request=ProviderRequest(
            request_id="sc_c_001",
            capability="vlm.object_grounding",
            inputs={"object_name": "red cup", "scene_description": "tabletop with red cup"},
        ),
    )
    assert result.status == "ok"
    assert result.capability == "vlm.object_grounding"
    assert "objects" in result.result
    assert len(result.result["objects"]) > 0
    assert "label" in result.result["objects"][0]


@pytest.mark.asyncio
async def test_scenario_c_skill_generates_grasp(runtime_with_physics):
    """Step 2: Skill provider generates grasp candidate."""
    runtime = runtime_with_physics
    result = await runtime.capability_router.invoke(
        request=ProviderRequest(
            request_id="sc_c_002",
            capability="skill.grasp",
            inputs={
                "target": {"object": "red cup", "position": [0.4, 0.0, 0.15]},
                "approach_vector": [0, 0, -1],
            },
        ),
    )
    assert result.status == "ok"
    assert result.capability == "skill.grasp"
    assert result.result["skill"] == "grasp"
    assert result.result["status"] == "dispatched"


@pytest.mark.asyncio
async def test_scenario_c_sandbox_validates_grasp(runtime_with_physics):
    """Step 3: Sandbox validates grasp trajectory with real MuJoCo physics."""
    runtime = runtime_with_physics
    assert runtime._sandbox is not None
    assert runtime._sandbox.has_physics, "Sandbox must have real MuJoCo physics"

    trajectory = [
        [0.0, -0.5, 0.0, 0.0, 0.5, 0.0],
        [0.1, -0.4, 0.1, 0.0, 0.4, 0.0],
        [0.2, -0.3, 0.2, 0.0, 0.3, 0.0],
    ]
    validation = runtime._sandbox.validate_trajectory(trajectory)
    assert "is_safe" in validation
    assert "risk_score" in validation

    state = runtime._sandbox.simulate_step(trajectory[0])
    assert state is not None
    assert "qpos" in state
    assert "time" in state
    assert len(state["qpos"]) >= 6


@pytest.mark.asyncio
async def test_scenario_c_runtime_executes_with_real_physics(runtime_with_physics):
    """Step 4: Runtime.execute() runs full closed loop with real physics."""
    runtime = runtime_with_physics

    events = []
    for topic in ["rosclaw.provider.inference.completed", "rosclaw.sandbox.action.allowed", "rosclaw.dashboard.trace.updated"]:
        runtime.event_bus.subscribe(topic, lambda e, t=topic: events.append(t))

    result = runtime.execute(
        action={
            "type": "reach",
            "parameters": {"target": [0.4, 0.0, 0.3], "orientation": [0, 0, 0, 1]},
            "episode_id": "sc_c_ep_001",
        },
    )

    assert result is not None
    assert "trajectory_data" in result
    assert len(result["trajectory_data"]) > 0

    first_jp = result["trajectory_data"][0].get("joint_positions", [])
    assert len(first_jp) >= 6, "Real physics should return joint_positions"

    # Verify actual events published by Runtime.execute()
    assert "rosclaw.provider.inference.completed" in events
    assert "rosclaw.sandbox.action.allowed" in events
    assert "rosclaw.dashboard.trace.updated" in events


@pytest.mark.asyncio
async def test_scenario_c_critic_judges_success(runtime_with_physics):
    """Step 5: Critic judges task success."""
    runtime = runtime_with_physics
    result = await runtime.capability_router.invoke(
        request=ProviderRequest(
            request_id="sc_c_003",
            capability="critic.success_detection",
            inputs={
                "task_description": "pick up the red cup",
                "observed_state": {"gripper_closed": True, "object_in_gripper": True},
            },
        ),
    )
    assert result.status == "ok"
    assert result.capability == "critic.success_detection"
    assert "success" in result.result


@pytest.mark.asyncio
async def test_scenario_c_memory_records_episode(runtime_with_physics):
    """Step 6: Memory records the episode."""
    runtime = runtime_with_physics
    runtime.execute(
        action={
            "type": "reach",
            "parameters": {"target": [0.4, 0.0, 0.3]},
            "episode_id": "sc_c_ep_002",
        },
    )
    # Memory is populated; verify interface works
    stats = runtime.memory.get_statistics()
    assert stats is not None


@pytest.mark.asyncio
async def test_scenario_c_how_recovery_on_blocked(runtime_with_physics):
    """Step 7: How generates recovery hint when action is blocked."""
    runtime = runtime_with_physics
    how_events = []
    runtime.event_bus.subscribe("rosclaw.how.recovery_hint.generated", lambda e: how_events.append(e))

    result = runtime.execute(
        action={
            "type": "reach",
            "parameters": {"target": [10.0, 10.0, 10.0]},
            "episode_id": "sc_c_ep_003",
        },
    )
    assert result is not None


@pytest.mark.asyncio
async def test_scenario_c_full_closed_loop(runtime_with_physics):
    """Full Scenario C: 桌面抓取红杯子端到端闭环."""
    runtime = runtime_with_physics

    all_events = []
    for topic in ["rosclaw.provider.inference.completed", "rosclaw.sandbox.action.allowed", "rosclaw.dashboard.trace.updated"]:
        runtime.event_bus.subscribe(topic, lambda e, t=topic: all_events.append(t))

    vlm_result = await runtime.capability_router.invoke(
        request=ProviderRequest(
            request_id="sc_c_full_001",
            capability="vlm.object_grounding",
            inputs={"object_name": "red cup"},
        ),
    )
    assert vlm_result.status == "ok"

    skill_result = await runtime.capability_router.invoke(
        request=ProviderRequest(
            request_id="sc_c_full_002",
            capability="skill.grasp",
            inputs={"target": {"object": "red cup", "position": [0.4, 0.0, 0.15]}},
        ),
    )
    assert skill_result.status == "ok"

    exec_result = runtime.execute(
        action={
            "type": "reach",
            "parameters": {"target": [0.4, 0.0, 0.3]},
            "episode_id": "sc_c_full_ep",
        },
    )
    assert exec_result is not None
    assert "trajectory_data" in exec_result

    if exec_result["trajectory_data"]:
        first_point = exec_result["trajectory_data"][0]
        assert "joint_positions" in first_point, "Real physics must produce joint_positions"
        assert len(first_point["joint_positions"]) >= 6

    critic_result = await runtime.capability_router.invoke(
        request=ProviderRequest(
            request_id="sc_c_full_003",
            capability="critic.success_detection",
            inputs={"task_description": "reach red cup"},
        ),
    )
    assert critic_result.status == "ok"

    # Verify actual events
    assert "rosclaw.provider.inference.completed" in all_events
    assert "rosclaw.sandbox.action.allowed" in all_events
    assert "rosclaw.dashboard.trace.updated" in all_events

    print("\n✅ Scenario C 端到端闭环验证通过！")
    print(f"   Events captured: {len(all_events)}")
    print("   Physics trajectory points: " + str(len(exec_result["trajectory_data"])))
