"""Tests for CapabilityClient composite task orchestration."""

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.provider.client import CapabilityClient


@pytest.fixture
def client():
    runtime = Runtime(RuntimeConfig(robot_id="test_bot", enable_provider=True))
    runtime.initialize()
    yield CapabilityClient(runtime.capability_router)
    runtime.stop()


@pytest.mark.asyncio
async def test_capability_client_pick_up_task(client):
    """Test 'pick up the red cup' decomposes into locate -> grasp -> verify."""
    result = await client.run_task(
        task="pick up the red cup",
        robot="ur5e",
        scene_input={"camera_topic": "/camera/color/image_raw"},
    )
    assert result.status == "success"
    assert len(result.steps) == 3
    assert result.steps[0]["capability"] == "vlm.object_grounding"
    assert result.steps[1]["capability"] == "skill.grasp"
    assert result.steps[2]["capability"] == "critic.success_detection"
    assert result.trace["total_latency_ms"] >= 0
    assert len(result.trace["steps"]) == 3


@pytest.mark.asyncio
async def test_capability_client_inspect_task(client):
    """Test 'what do you see' maps to scene_understanding only."""
    result = await client.run_task(
        task="what do you see on the table?",
        robot="ur5e",
        scene_input={"camera_topic": "/camera/color/image_raw"},
    )
    assert result.status == "success"
    assert len(result.steps) == 1
    assert result.steps[0]["capability"] == "vlm.scene_understanding"


@pytest.mark.asyncio
async def test_capability_client_place_task(client):
    """Test 'place the cup into the bin' maps to pick_and_place + verify."""
    result = await client.run_task(
        task="place the red cup into the blue bin",
        robot="ur5e",
        scene_input={"camera_topic": "/camera/color/image_raw"},
    )
    assert result.status == "success"
    assert len(result.steps) == 2
    assert result.steps[0]["capability"] == "skill.pick_and_place"
    assert result.steps[1]["capability"] == "critic.success_detection"


@pytest.mark.asyncio
async def test_capability_client_locate_and_grasp(client):
    """Test convenience method locate_and_grasp."""
    result = await client.locate_and_grasp(
        object_name="screwdriver",
        robot="ur5e",
        camera_topic="/camera/color/image_raw",
    )
    assert result.status == "success"
    assert result.steps[0]["capability"] == "vlm.object_grounding"
    assert result.steps[1]["capability"] == "skill.grasp"


@pytest.mark.asyncio
async def test_capability_client_task_result_fields(client):
    """TaskResult contains all expected fields."""
    result = await client.run_task(
        task="pick up the red cup",
        robot="ur5e",
        scene_input={"camera_topic": "/camera/color/image_raw"},
    )
    assert result.task == "pick up the red cup"
    assert result.status in ("success", "partial", "failed")
    assert isinstance(result.steps, list)
    assert isinstance(result.trace, dict)
    assert "trace_id" in result.trace
    assert "total_latency_ms" in result.trace
    assert isinstance(result.final_result, dict)
    assert isinstance(result.errors, list)


@pytest.mark.asyncio
async def test_capability_client_failed_step_aborts(client):
    """If a critical step fails, subsequent steps are skipped."""
    # Force a failure by requesting an unsupported capability
    result = await client.run_task(
        task="navigate to the kitchen",  # mock provider doesn't support skill.navigate
        robot="ur5e",
        scene_input={},
    )
    assert result.status == "failed"
    # Should have attempted navigate step and then aborted
    assert len(result.steps) >= 1
    assert any(s["status"] == "failed" for s in result.steps)
