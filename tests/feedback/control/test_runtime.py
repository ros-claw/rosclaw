from __future__ import annotations

import math
from collections.abc import Iterator, Mapping

import pytest

from rosclaw.feedback.contracts import FallbackMode, FeedbackFrame, FeedbackLoopSpec
from rosclaw.feedback.controllers.pid import PIDGains, PIDResidualController
from rosclaw.feedback.replay import RecordedLatencyClock, verify_feedback_replay
from rosclaw.feedback.runtime import FeedbackRuntime


def _runtime(
    *,
    clock=None,
    deadline_ms: float = 5.0,
    deadline_fallback: FallbackMode = FallbackMode.BASE_POLICY_ONLY,
) -> FeedbackRuntime:
    controller = PIDResidualController(
        {"roll": PIDGains(kp=0.5, kd=0.01)},
        {"roll": "joint:waist_roll_joint"},
    )
    spec = FeedbackLoopSpec(
        loop_id="test/pid",
        body_hash="sha256:" + "1" * 64,
        controller_hash=controller.controller_hash,
        reference_signals=("roll",),
        observation_signals=("roll",),
        output_limits={"joint:waist_roll_joint": 0.1},
        deadline_ms=deadline_ms,
        fallback_deadline_miss=deadline_fallback,
    )
    kwargs = {"spec": spec, "controller": controller}
    if clock is not None:
        kwargs["compute_clock_ns"] = clock
    return FeedbackRuntime(**kwargs)  # type: ignore[arg-type]


def _tick(runtime: FeedbackRuntime, index: int, actual: float):
    timestamp = 1_000_000_000 + index * 5_000_000
    return runtime.tick(
        timestamp_ns=timestamp,
        observation_timestamp_ns=timestamp,
        phase=0.5,
        reference={"roll": 0.0},
        actual={"roll": actual},
        base_action={"joint:waist_roll_joint": 0.2},
    )


def test_runtime_computes_clamps_and_records_without_event_bus() -> None:
    # This test checks projection/recording; deadline behavior has separate tests.
    runtime = _runtime(clock=RecordedLatencyClock((100_000,)))
    command = _tick(runtime, 0, actual=1.0)

    assert command.projected == {"joint:waist_roll_joint": -0.1}
    assert command.saturation_count == 1
    assert command.deadline_met
    assert runtime.records[0].latency_ns == 100_000
    error, _ = runtime.recent_error(1)
    residual, _ = runtime.recent_residual(1)
    assert error.tolist() == [[-1.0]]
    assert residual.tolist() == [[-0.1]]


def test_stale_observation_fails_closed_without_poisoning_estimator() -> None:
    runtime = _runtime()
    command = runtime.tick(
        timestamp_ns=1_100_000_000,
        observation_timestamp_ns=1_000_000_000,
        phase=0.5,
        reference={"roll": 0.0},
        actual={"roll": 0.3},
        base_action={"joint:waist_roll_joint": 0.2},
    )
    recovered = _tick(runtime, 21, actual=0.2)

    assert command.projected == {}
    assert command.fallback is not None
    assert command.reasons[0].startswith("stale_observation")
    assert recovered.projected


def test_deadline_miss_discards_computed_residual() -> None:
    readings: Iterator[int] = iter((0, 2_000_000))
    runtime = _runtime(clock=lambda: next(readings), deadline_ms=1.0)
    command = _tick(runtime, 0, actual=0.1)

    assert command.projected == {}
    assert not command.deadline_met
    assert command.reasons[0].startswith("deadline_miss")


def test_freeze_deadline_fallback_holds_last_safe_projected_residual() -> None:
    readings: Iterator[int] = iter((0, 100_000, 100_000, 2_100_000))
    runtime = _runtime(
        clock=lambda: next(readings),
        deadline_ms=1.0,
        deadline_fallback=FallbackMode.FREEZE_AND_STABILIZE,
    )
    safe = _tick(runtime, 0, actual=0.1)
    missed = _tick(runtime, 1, actual=0.4)

    assert missed.projected == safe.projected
    assert missed.fallback is FallbackMode.FREEZE_AND_STABILIZE
    assert not missed.deadline_met


def test_invalid_controller_output_fails_closed() -> None:
    class InvalidController:
        @property
        def controller_hash(self) -> str:
            return "sha256:" + "2" * 64

        def reset(self) -> None:
            pass

        def compute(
            self, frame: FeedbackFrame, base_action: Mapping[str, float]
        ) -> Mapping[str, float]:
            del frame, base_action
            return {"not-allowed": 1.0}

        def config_dict(self) -> dict[str, object]:
            return {"controller_type": "invalid"}

    controller = InvalidController()
    spec = FeedbackLoopSpec(
        loop_id="test/invalid",
        body_hash="sha256:" + "1" * 64,
        controller_hash=controller.controller_hash,
        reference_signals=("roll",),
        observation_signals=("roll",),
        output_limits={"joint": 0.1},
    )
    runtime = FeedbackRuntime(spec=spec, controller=controller)
    command = runtime.tick(
        timestamp_ns=1,
        observation_timestamp_ns=1,
        phase=0.5,
        reference={"roll": 0.0},
        actual={"roll": 0.1},
        base_action={"joint": 0.0},
    )

    assert command.projected == {}
    assert command.reasons[0].startswith("unsafe_projection")


@pytest.mark.parametrize("mode", ("nonfinite", "exception"))
def test_nonfinite_or_crashing_controller_fails_closed(mode: str) -> None:
    class FaultyController:
        @property
        def controller_hash(self) -> str:
            return "sha256:" + "3" * 64

        def reset(self) -> None:
            pass

        def compute(
            self, frame: FeedbackFrame, base_action: Mapping[str, float]
        ) -> Mapping[str, float]:
            del frame, base_action
            if mode == "exception":
                raise RuntimeError("controller crashed")
            return {"joint": float("nan")}

        def config_dict(self) -> dict[str, object]:
            return {"controller_type": "faulty"}

    controller = FaultyController()
    spec = FeedbackLoopSpec(
        loop_id="test/faulty",
        body_hash="sha256:" + "1" * 64,
        controller_hash=controller.controller_hash,
        reference_signals=("roll",),
        observation_signals=("roll",),
        output_limits={"joint": 0.1},
    )
    runtime = FeedbackRuntime(spec=spec, controller=controller)
    command = runtime.tick(
        timestamp_ns=1,
        observation_timestamp_ns=1,
        phase=0.5,
        reference={"roll": 0.0},
        actual={"roll": 0.1},
        base_action={"joint": 0.0},
    )

    assert command.projected == {}
    assert command.fallback is not None
    assert command.raw == {}


def test_all_stale_trace_still_builds_content_addressed_receipt() -> None:
    runtime = _runtime()
    runtime.tick(
        timestamp_ns=100_000_000,
        observation_timestamp_ns=0,
        phase=0.5,
        reference={"roll": 0.0},
        actual={"roll": 0.1},
        base_action={"joint:waist_roll_joint": 0.0},
    )

    receipt = runtime.build_receipt(action_id="stale", strict_replay=False)

    assert receipt.initial_error_rms is None
    assert receipt.final_error_rms is None
    assert not receipt.tracking_improved
    assert receipt.receipt_hash.startswith("sha256:")


def test_strict_replay_and_receipt_are_content_bound() -> None:
    runtime = _runtime()
    for index, actual in enumerate((0.3, 0.2, 0.1, 0.05)):
        _tick(runtime, index, actual)
    expected = [record.command_hash for record in runtime.records]
    inputs = [record.input for record in runtime.records]
    report = verify_feedback_replay(_runtime, inputs, expected)
    receipt = runtime.build_receipt(
        action_id="action-test",
        strict_replay=report.strict_replay,
        evidence_domain="SIM",
    )

    assert report.strict_replay
    assert receipt.strict_replay
    assert receipt.tracking_improved
    assert receipt.deadline_miss_count == 0
    assert receipt.evidence_domain == "SIM"
    assert receipt.receipt_hash.startswith("sha256:")


def test_runtime_rejects_controller_spec_mismatch() -> None:
    controller = PIDResidualController({"roll": PIDGains(kp=1.0)})
    spec = FeedbackLoopSpec(
        loop_id="test/mismatch",
        body_hash="sha256:" + "1" * 64,
        controller_hash="sha256:" + "f" * 64,
        reference_signals=("roll",),
        observation_signals=("roll",),
        output_limits={"roll": 0.1},
    )
    with pytest.raises(ValueError, match="controller hash"):
        FeedbackRuntime(spec=spec, controller=controller)


def test_recorded_latency_clock_replays_start_end_pairs() -> None:
    clock = RecordedLatencyClock((100, 250))

    assert (clock(), clock(), clock(), clock()) == (0, 100, 100, 350)
    with pytest.raises(RuntimeError, match="exhausted"):
        clock()


def test_non_finite_input_fails_closed_instead_of_raising() -> None:
    runtime = _runtime()
    command = runtime.tick(
        timestamp_ns=1_000_000_000,
        observation_timestamp_ns=1_000_000_000,
        phase=0.5,
        reference={"roll": 0.0},
        actual={"roll": float("nan")},
        base_action={"joint:waist_roll_joint": float("inf")},
    )
    recovered = _tick(runtime, 1, actual=0.2)

    assert command.fallback is not None
    assert command.reasons[0].startswith("invalid_input")
    assert all(math.isfinite(value) for value in command.projected.values())
    assert math.isfinite(runtime.records[-2].residual_norm)
    assert recovered.fallback is None
    assert recovered.projected


def test_non_finite_base_action_does_not_leak_into_fallback_base() -> None:
    runtime = _runtime()
    command = runtime.tick(
        timestamp_ns=1_000_000_000,
        observation_timestamp_ns=1_000_000_000,
        phase=0.5,
        reference={"roll": 0.0},
        actual={"roll": 0.3},
        base_action={"joint:waist_roll_joint": float("nan")},
    )

    assert command.fallback is not None
    assert all(math.isfinite(value) for value in command.projected.values())
    recorded = runtime.records[-1]
    assert all(math.isfinite(value) for value in recorded.input.base_action.values())
