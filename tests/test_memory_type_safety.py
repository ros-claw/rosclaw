"""
Test type safety for EmbodiedMemory proxy methods.

Verifies that Protocol-based type annotations are applied correctly
and that the conditional import pattern works as expected.

P1 Issue 1: https://github.com/ros-claw/rosclaw-v1.0/issues/XXX
"""

import sys
from typing import Any
from unittest.mock import MagicMock
import pytest

# Import MemoryInterface directly to avoid triggering rosclaw.__init__
# which imports all modules (including MCPHub which has unrelated issues)
sys.path.insert(0, '/home/ubuntu/rosclaw/rosclaw_memory/powermem/src')
from rosclaw.memory.interface import MemoryInterface  # noqa: E402


@pytest.fixture(autouse=True)
def reload_memory_interface():
    """Ensure rosclaw.memory.interface is in clean state before each test."""
    import importlib
    import rosclaw.memory.interface
    importlib.reload(rosclaw.memory.interface)


def test_protocol_import_with_powermem():
    """Verify Protocol types are imported when powermem is available."""
    # powermem is in the test environment, so protocols should be imported
    from rosclaw.memory.interface import (
        _HAS_POWERMEM_PROTOCOLS,
        WorldObjectLike,
        PoseLike,
        Vec3Like,
        TemporalIntervalLike,
        PermanenceReportLike,
        MemoryAtomLike,
    )

    assert _HAS_POWERMEM_PROTOCOLS is True

    # Verify they are Protocol types (not Any)
    from powermem.embodied.protocols import (
        WorldObjectLike as RealWorldObjectLike,
        PoseLike as RealPoseLike,
        Vec3Like as RealVec3Like,
        TemporalIntervalLike as RealTemporalIntervalLike,
        PermanenceReportLike as RealPermanenceReportLike,
        MemoryAtomLike as RealMemoryAtomLike,
    )

    assert WorldObjectLike is RealWorldObjectLike
    assert PoseLike is RealPoseLike
    assert Vec3Like is RealVec3Like
    assert TemporalIntervalLike is RealTemporalIntervalLike
    assert PermanenceReportLike is RealPermanenceReportLike
    assert MemoryAtomLike is RealMemoryAtomLike


def test_protocol_import_without_powermem(monkeypatch):
    """Verify fallback to Any when powermem is not available."""
    # Simulate powermem not being installed
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("powermem"):
            raise ImportError("powermem not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Reload the module to trigger the fallback
    import importlib
    import rosclaw.memory.interface
    importlib.reload(rosclaw.memory.interface)

    from rosclaw.memory.interface import _HAS_POWERMEM_PROTOCOLS
    assert _HAS_POWERMEM_PROTOCOLS is False

    # Restore original import and reload module to restore original state
    monkeypatch.setattr(builtins, "__import__", original_import)

    # IMPORTANT: Reload the module again to restore original state for subsequent tests
    # Without this, subsequent tests see the reloaded module with _HAS_POWERMEM_PROTOCOLS=False
    importlib.reload(rosclaw.memory.interface)


def test_proxy_methods_have_type_annotations():
    """Verify proxy methods use Protocol types instead of Any."""
    from typing import get_type_hints

    # Re-import MemoryInterface from the freshly reloaded module
    from rosclaw.memory.interface import MemoryInterface as FreshMemoryInterface

    # Get type hints for proxy methods
    get_type_hints(FreshMemoryInterface)

    # Verify add_world_object uses WorldObjectLike (not Any)
    add_hints = get_type_hints(FreshMemoryInterface.add_world_object)
    obj_type = add_hints.get("obj")
    assert obj_type is not None
    assert obj_type is not Any, "add_world_object should use Protocol type, not Any"
    assert "WorldObjectLike" in str(obj_type)

    # Verify get_world_object returns Optional[WorldObjectLike]
    get_hints = get_type_hints(FreshMemoryInterface.get_world_object)
    assert "WorldObjectLike" in str(get_hints.get("return", ""))

    # Verify update_world_object_pose uses PoseLike (not Any)
    update_hints = get_type_hints(FreshMemoryInterface.update_world_object_pose)
    pose_type = update_hints.get("pose")
    assert pose_type is not None
    assert pose_type is not Any, "update_world_object_pose should use Protocol type, not Any"
    assert "PoseLike" in str(pose_type)

    # Verify search_world_objects uses Vec3Like (not Any)
    search_hints = get_type_hints(FreshMemoryInterface.search_world_objects)
    center_type = search_hints.get("center")
    assert center_type is not None
    assert center_type is not Any, "search_world_objects should use Protocol type, not Any"
    assert "Vec3Like" in str(center_type)

    # Verify sync_scene_objects uses PermanenceReportLike
    sync_hints = get_type_hints(FreshMemoryInterface.sync_scene_objects)
    # Note: Optional[PermanenceReportLike] is Union[PermanenceReportLike, None]
    assert "PermanenceReportLike" in str(sync_hints.get("return", ""))

    # Verify record_trajectory uses Vec3Like
    record_hints = get_type_hints(FreshMemoryInterface.record_trajectory)
    # Note: list[tuple[Vec3Like, float]] is a complex type
    assert "Vec3Like" in str(record_hints.get("waypoints", ""))

    # Verify cognitive_search uses Vec3Like, TemporalIntervalLike, and MemoryAtomLike
    cognitive_hints = get_type_hints(FreshMemoryInterface.cognitive_search)
    spatial_type = cognitive_hints.get("spatial_center")
    assert spatial_type is not None
    assert "Vec3Like" in str(spatial_type)
    temporal_type = cognitive_hints.get("temporal_interval")
    assert temporal_type is not None
    assert "TemporalIntervalLike" in str(temporal_type)
    # Note: list[MemoryAtomLike] is a complex type
    assert "MemoryAtomLike" in str(cognitive_hints.get("return", ""))


def test_proxy_methods_with_embodied_memory():
    """Verify proxy methods work correctly with EmbodiedMemory."""
    # Create a mock EmbodiedMemory
    mock_embodied = MagicMock()
    mock_embodied.add_world_object.return_value = "obj_123"
    mock_embodied.get_world_object.return_value = MagicMock()
    mock_embodied.update_world_object_pose.return_value = True
    mock_embodied.search_world_objects.return_value = []
    mock_embodied.get_scene_graph.return_value = ([], [])
    mock_embodied.compute_relations.return_value = []
    mock_embodied.sync_scene_objects.return_value = MagicMock()
    mock_embodied.record_trajectory.return_value = 42
    mock_embodied.search_similar_trajectories.return_value = []
    mock_embodied.cognitive_search.return_value = []

    # Create MemoryInterface with mock EmbodiedMemory
    mem = MemoryInterface(robot_id="test_bot", embodied_memory=mock_embodied)

    # Test add_world_object
    mock_obj = MagicMock()
    result = mem.add_world_object(mock_obj)
    assert result == "obj_123"
    mock_embodied.add_world_object.assert_called_once_with(mock_obj)

    # Test get_world_object
    mem.get_world_object("obj_123")
    mock_embodied.get_world_object.assert_called_once_with("obj_123")

    # Test update_world_object_pose
    mock_pose = MagicMock()
    mem.update_world_object_pose("obj_123", mock_pose, state="moved")
    mock_embodied.update_world_object_pose.assert_called_once_with("obj_123", mock_pose, "moved")

    # Test search_world_objects
    mock_center = MagicMock()
    mem.search_world_objects(mock_center, radius=2.0, scene_id="scene_1")
    mock_embodied.search_world_objects.assert_called_once_with(mock_center, 2.0, "scene_1")

    # Test get_scene_graph
    mem.get_scene_graph("scene_1")
    mock_embodied.get_scene_graph.assert_called_once_with("scene_1")

    # Test compute_relations
    mem.compute_relations("scene_1", spatial_tolerance=0.1)
    mock_embodied.compute_relations.assert_called_once_with("scene_1", 0.1)

    # Test sync_scene_objects
    mock_detections = [MagicMock()]
    mem.sync_scene_objects("scene_1", mock_detections, timestamp_sec=123.0, occlusion_radius=0.5)
    mock_embodied.sync_scene_objects.assert_called_once_with(
        "scene_1", mock_detections, 123.0, 0.5
    )

    # Test record_trajectory
    mock_waypoints = [(MagicMock(), 1.0), (MagicMock(), 2.0)]
    result = mem.record_trajectory("move to goal", mock_waypoints)
    assert result == 42
    mock_embodied.record_trajectory.assert_called_once_with("move to goal", mock_waypoints)

    # Test search_similar_trajectories
    mem.search_similar_trajectories(mock_waypoints, top_k=5, max_dtw_distance=0.5)
    mock_embodied.search_similar_trajectories.assert_called_once_with(
        mock_waypoints, 5, 0.5
    )

    # Test cognitive_search
    mock_center = MagicMock()
    mock_interval = MagicMock()
    mem.cognitive_search(
        "test query",
        spatial_center=mock_center,
        spatial_radius=2.0,
        temporal_interval=mock_interval,
        limit=10,
    )
    mock_embodied.cognitive_search.assert_called_once_with(
        query="test query",
        limit=10,
        spatial_center=mock_center,
        spatial_radius=2.0,
        temporal_interval=mock_interval,
    )


def test_proxy_methods_without_embodied_memory():
    """Verify proxy methods return safe defaults when EmbodiedMemory is not attached."""
    mem = MemoryInterface(robot_id="test_bot", embodied_memory=None)

    # All proxy methods should return None/[]/False
    assert mem.add_world_object(MagicMock()) is None
    assert mem.get_world_object("obj_123") is None
    assert mem.update_world_object_pose("obj_123", MagicMock()) is False
    assert mem.search_world_objects(MagicMock(), 2.0) == []
    assert mem.get_scene_graph("scene_1") == ([], [])
    assert mem.compute_relations("scene_1") == []
    assert mem.sync_scene_objects("scene_1", [], 123.0) is None
    assert mem.record_trajectory("test", []) is None
    assert mem.search_similar_trajectories([]) == []
    assert mem.cognitive_search("test") == []
    assert mem.run_meditation() == {"success": False, "error": "EmbodiedMemory not attached"}


def test_protocol_runtime_checking():
    """Verify Protocol types support runtime isinstance() checks."""
    from rosclaw.memory.interface import (
        WorldObjectLike, Vec3Like, PoseLike, _HAS_POWERMEM_PROTOCOLS
    )

    # Skip this test when powermem is not available (WorldObjectLike would be Any)
    if not _HAS_POWERMEM_PROTOCOLS:
        pytest.skip("powermem not available - Protocol types fall back to Any")

    # Create a mock object that satisfies WorldObjectLike protocol
    mock_obj = MagicMock()
    mock_obj.obj_id = "obj_123"
    mock_obj.obj_type = "box"
    mock_obj.name = "Test Box"
    mock_obj.pose = MagicMock()
    mock_obj.pose.position = MagicMock()
    mock_obj.pose.position.x = 1.0
    mock_obj.pose.position.y = 2.0
    mock_obj.pose.position.z = 0.0
    mock_obj.pose.orientation = MagicMock()
    mock_obj.pose.orientation.w = 1.0
    mock_obj.pose.orientation.x = 0.0
    mock_obj.pose.orientation.y = 0.0
    mock_obj.pose.orientation.z = 0.0
    mock_obj.scene_id = "scene_1"
    mock_obj.state = "present"
    mock_obj.occlusion_status = "visible"
    mock_obj.confidence = 1.0
    mock_obj.last_seen_sec = 0.0
    mock_obj.semantic_tags = []
    mock_obj.to_dict = MagicMock(return_value={"obj_id": "obj_123"})

    # Protocol should support isinstance() check (runtime_checkable)
    assert isinstance(mock_obj, WorldObjectLike)
    assert isinstance(mock_obj.pose.position, Vec3Like)
    assert isinstance(mock_obj.pose, PoseLike)
