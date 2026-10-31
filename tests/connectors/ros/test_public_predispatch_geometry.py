"""Known undispatched geometry rejection consumes attempts, never safety evidence."""

from dataclasses import replace

import pytest

from rosclaw.kernel import ActionState
from tests.connectors.ros.test_public_coverage_execution import setup_executor


def changed_after_enable(monkeypatch, fault="geometry"):
    _, executor, port, action = setup_executor()
    original = port.execute
    proposals = []

    def choose(*args, **kwargs):
        proposals.append(1)
        x, y = port.xy
        return {
            "kind": "WORK",
            "path_xy": [[x, y], [3.2, 2.0] if len(proposals) == 1 else [x, y + 0.7]],
        }

    changed = False

    def execute(capability, *args, **kwargs):
        nonlocal changed
        result = original(capability, *args, **kwargs)
        if capability == "cleaning.enable" and not changed:
            changed = True
            if fault == "entry_pose_shift":
                port.xy = (port.xy[0] + 0.12, port.xy[1])
            elif fault == "geometry":
                port.grid.free[20, 32] = False
            elif fault == "unknown_native":
                port.motion_settled = False
            elif fault == "unknown_geometry":

                def unavailable(*args, **kwargs):
                    raise ValueError("Unknown public geometry")

                monkeypatch.setattr(port.grid, "motion_path_safe", unavailable)
            else:
                port.fault = fault
        return result

    monkeypatch.setattr(port, "execute", execute)
    monkeypatch.setattr(executor.ledger, "next_work", choose)
    return executor, port, action, proposals


@pytest.mark.parametrize("fault", ["geometry", "entry_pose_shift"])
def test_new_obstacle_after_brush_enable_replans_before_native_send(monkeypatch, fault):
    executor, port, action, proposals = changed_after_enable(monkeypatch, fault)
    deadline = action.deadline_at
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    assert len(proposals) == 2
    assert result.dispatch_result["completed_work"] == 1
    assert result.dispatch_result["completed_continuous_strokes"] == 1
    events = result.dispatch_result["observed_post_entry_recoveries"]
    assert len(events) == 1 and events[0]["original_native_terminal_known"] is True
    assert events[0]["fresh_stopped_off"] is True
    assert events[0]["rejection_phase"] == "before_native_path_dispatch"
    assert events[0]["cleaner_enabled_at_rejection"] is True
    assert events[0]["native_goal_dispatched"] is False
    assert events[0]["rejected_public_path"]["entry_limit_m"] == 0.10
    if fault == "entry_pose_shift":
        assert events[0]["rejected_public_path"]["entry_distance_m"] > 0.10
    assert sum(call[0] == "navigation.follow_path" for call in port.calls) == 1
    assert sum(call[0] == "cleaning.enable" for call in port.calls) == 2
    assert action.deadline_at == deadline
    assert port.stop_calls == 0 and not port.brush


def test_original_work_attempt_limit_is_not_extended_by_predispatch_change(monkeypatch):
    executor, port, action, proposals = changed_after_enable(monkeypatch)
    result = executor(replace(action, arguments={"maximum_work": 1}))
    assert result.final_state is ActionState.FAILED
    assert len(proposals) == 1
    assert result.dispatch_result["completed_work"] == 0
    assert result.dispatch_result["completed_continuous_strokes"] == 0
    assert len(result.dispatch_result["observed_post_entry_recoveries"]) == 1
    assert not any(call[0] == "navigation.follow_path" for call in port.calls)
    assert executor.stopping.is_set() and not port.brush


@pytest.mark.parametrize("fault", ["depth", "body", "epoch", "unknown_native", "unknown_geometry"])
def test_unsafe_or_unknown_changes_cannot_use_geometry_replanning(monkeypatch, fault):
    executor, port, action, proposals = changed_after_enable(monkeypatch, fault)
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert len(proposals) == 1
    assert result.dispatch_result["observed_post_entry_recoveries"] == []
    assert not any(call[0] == "navigation.follow_path" for call in port.calls)
    assert executor.stopping.is_set() and port.stop_calls == 1
