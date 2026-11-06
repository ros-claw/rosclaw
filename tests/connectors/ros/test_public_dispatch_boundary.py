"""Software ports only: last dispatch guards retain original work and bounds."""

import time

import numpy as np
import pytest

from tests.connectors.ros.test_public_coverage_execution import setup_executor


def ready_executor():
    _, executor, port, action = setup_executor()
    executor._parent = action
    executor._deadline = time.monotonic() + 20.0
    return executor, port, action


@pytest.mark.parametrize("capability", ["cleaning.enable", "navigation.follow_path"])
@pytest.mark.parametrize(
    "fault", ["depth", "body", "epoch", "publisher", "low_soc", "docked", "native_unknown"]
)
def test_intent_persistence_cannot_authorize_a_later_invalid_dispatch(capability, fault):
    executor, port, _ = ready_executor()
    original_deadline = executor._deadline

    def persist(*args):
        if fault == "native_unknown":
            port.motion_settled = False
        else:
            port.fault = fault

    port.record_primitive_intent = persist
    arguments = (
        {"path_xy": [[2.0, 2.0], [2.4, 2.0]]} if capability.startswith("navigation.") else {}
    )
    with pytest.raises((RuntimeError, ValueError)):
        executor._primitive(capability, arguments)
    assert port.calls == [], "A post-dispatch failure cannot undo an unsafe command"
    assert not port.brush and executor._deadline == original_deadline


@pytest.mark.parametrize("change", ["entry", "footprint"])
def test_path_is_rechecked_after_intent_persistence(change):
    executor, port, _ = ready_executor()

    def persist(*args):
        if change == "entry":
            port.xy = (2.0, 2.3)
        else:
            port.grid.occupancy_space.free[20, 24] = False

    port.record_primitive_intent = persist
    with pytest.raises((RuntimeError, ValueError)):
        executor._primitive("navigation.follow_path", {"path_xy": [[2.0, 2.0], [2.4, 2.0]]})
    assert port.calls == [] and not port.brush


def test_depth_fault_does_not_remove_the_original_brush_disable_path():
    executor, port, _ = ready_executor()
    port.brush = True
    port.record_primitive_intent = lambda *args: setattr(port, "fault", "depth")
    executor._primitive("cleaning.disable", {})
    assert [call[0] for call in port.calls] == ["cleaning.disable"]
    assert not port.brush


def test_persistence_does_not_receive_mutable_dispatch_arguments():
    executor, port, _ = ready_executor()
    arguments = {"path_xy": [[2.0, 2.0], [2.4, 2.0]]}

    def persist(sample, capability, saved, *args):
        saved["path_xy"][-1] = [2.0, 2.4]

    port.record_primitive_intent = persist
    executor._primitive("navigation.follow_path", arguments)
    assert port.xy == pytest.approx((2.4, 2.0))
    assert arguments == {"path_xy": [[2.0, 2.0], [2.4, 2.0]]}


def test_remaining_work_budget_is_rechecked_after_brush_intent_persistence():
    executor, port, _ = ready_executor()
    persisted = []
    checks = []

    def intent(sample, capability, *args):
        persisted.append(capability)

    def admit(phase, *args, **kwargs):
        checks.append(phase)
        if phase == "before_brush_enable" and "cleaning.enable" in persisted:
            raise TimeoutError("Original work no longer fits after intent persistence")

    port.record_primitive_intent = intent
    port.admit_finite_budget = admit
    with pytest.raises(TimeoutError, match="Original work no longer fits"):
        executor._execute_work({"path_xy": [[2.0, 2.0], [2.4, 2.0]]})
    assert checks.count("before_brush_enable") == 2
    assert port.calls == [] and not port.brush


@pytest.mark.parametrize(
    "replacement",
    [
        [[2.4, 2.0], [2.4, 3.5]],
        [[2.4, 2.0], [2.8, 2.0]],
        [[2.41, 2.0], [2.5, 2.0]],
        [[2.4, 2.0], [2.4, 2.1]],
    ],
)
def test_exact_transfer_budget_callback_cannot_replace_unissued_cleaning(replacement):
    executor, port, _ = ready_executor()
    original = np.array([[2.4, 2.0], [2.6, 2.0]])
    held = original.copy()
    port.prepare_finite_budget_candidate = lambda *args, **kwargs: np.asarray(replacement)
    with pytest.raises(ValueError, match="original unissued prefix"):
        executor._execute_work({"path_xy": original.tolist()})
    assert port.calls == [] and not port.brush
    np.testing.assert_array_equal(original, held)


@pytest.mark.parametrize("length", [0.1, 0.2])
def test_exact_transfer_budget_retains_a_legal_shorter_prefix(length):
    executor, port, _ = ready_executor()
    original = np.array([[2.4, 2.0], [2.6, 2.0]])
    port.prepare_finite_budget_candidate = lambda *args, **kwargs: np.array(
        [[2.4, 2.0], [2.4 + length, 2.0]]
    )
    assert executor._execute_work({"path_xy": original.tolist()}) is True
    assert not port.brush
    assert port.xy == pytest.approx((2.4 + length, 2.0))


def test_exact_budget_shortening_cannot_complete_an_original_continuous_stroke_goal():
    from dataclasses import replace

    from rosclaw.connectors.ros.public_coverage.coverage_variants import prefix
    from rosclaw.kernel import ActionState

    _, executor, port, action = setup_executor()
    original = np.array([[2.4, 2.0], [3.0, 2.0]])
    port.begin_finite_budget_selection = lambda parent: None
    port.prepare_finite_budget_candidate = lambda path, *args, exact_transfer=False: (
        prefix(path, 0.05) if exact_transfer else path
    )

    def select_work(*args, **kwargs):
        assert kwargs["require_continuous_stroke"] is True
        path = kwargs["prepare_candidate"](original.copy(), 0.4)
        return {"kind": "WORK", "path_xy": path.tolist()}

    executor.ledger.next_work = select_work
    result = executor(
        replace(
            action,
            arguments={
                "maximum_work": 1,
                "minimum_completed_work": 1,
                "minimum_completed_strokes": 1,
            },
        )
    )
    assert result.final_state is ActionState.FAILED
    assert result.dispatch_result["completed_continuous_strokes"] == 0
    assert not any(call[0] == "cleaning.enable" for call in port.calls)
    assert not port.brush
