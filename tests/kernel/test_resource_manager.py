"""Exclusive resource lease contracts."""

from __future__ import annotations

from rosclaw.kernel import ResourceManager


def test_exclusive_lease_blocks_second_action_until_release() -> None:
    manager = ResourceManager()
    first = manager.acquire("left_arm", "action-1", timeout_sec=0.01)
    assert first is not None

    blocked = manager.acquire("left_arm", "action-2", timeout_sec=0.001)
    assert blocked is None
    assert manager.active_lease("left_arm") == first.lease

    first.release()
    second = manager.acquire("left_arm", "action-2", timeout_sec=0.01)
    assert second is not None
    second.release()
    assert manager.active_lease("left_arm") is None


def test_lease_release_is_idempotent() -> None:
    manager = ResourceManager()
    handle = manager.acquire("camera_front", "observe-1", timeout_sec=0.01)
    assert handle is not None

    handle.release()
    handle.release()

    assert manager.active_lease("camera_front") is None
