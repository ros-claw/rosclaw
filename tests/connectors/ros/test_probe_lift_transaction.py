"""Synthetic retained-source lift races; no service, World, Node or actuator."""

import importlib

import pytest

from tests.connectors.ros.test_closed_backend_probe import Tape, retained
from tests.connectors.ros.test_closed_backend_probe import module as source_module

module = source_module


def start(tape, transaction="a" * 64):
    request = importlib.import_module("probe_lift_evidence").lift_request(tape.policy)
    tape.emit(
        "backend_probe_lift_begin",
        {
            "transaction_id": transaction,
            "request": retained(request, "gz.msgs.Pose_protobuf_text"),
            "received_monotonic_sec": 100 + tape.i * 0.05 + 0.003,
            "received_unix_ns": tape.unix + round(tape.i * 0.05 * 1e9) + 3_000_000,
        },
    )


def ack(tape, transaction="a" * 64, response=b"data: true\n"):
    tape.emit(
        "backend_probe_lift_ack",
        {
            "transaction_id": transaction,
            "request": retained(
                importlib.import_module("probe_lift_evidence").lift_request(tape.policy),
                "gz.msgs.Pose_protobuf_text",
            ),
            "response": retained(response, "gz.msgs.Boolean_protobuf_text"),
            "returncode": 0,
            "received_monotonic_sec": 100 + tape.i * 0.05 + 0.003,
            "received_unix_ns": tape.unix + round(tape.i * 0.05 * 1e9) + 3_000_000,
        },
    )


@pytest.fixture
def ground(module, tmp_path):
    tape = Tape(module, tmp_path / "source.jsonl")
    for _ in range(12):
        tape.frame()
    assert tape.replay.tracker.phase == "READY_FOR_LIFT"
    return tape


def test_original_lift_frames_before_reply_are_retained_then_measured_after_reply(ground):
    tape = ground
    start(tape)
    for _ in range(2):
        tape.frame(z=10, touching=False)
    assert len(tape.replay.pending) == 2
    assert tape.replay.tracker.phase == "READY_FOR_LIFT"
    assert tape.replay.tracker.cycles == 0
    original = list(tape.replay.pending)
    ack(tape)
    assert not tape.replay.pending and tape.replay.lift_transaction is None
    assert tape.replay.tracker.phase == "WAIT_CLEAR_AFTER_LIFT"
    assert not tape.replay.tracker.lift_pose_confirmed
    assert tape.replay.tracker.previous[3] == original[-1][2] - 1_000_000
    tape.frame(z=9.99, touching=False)
    for _ in range(11):
        tape.frame(z=9, touching=False)
    for _ in range(12):
        tape.frame()
    assert tape.replay.tracker.cycles == 1
    tape.write()
    review = tape.module.closed_probe_replay(
        tape.path,
        tape.policy,
        plugin_path=tape.path.parent / "unused",
        pose_frame="actual_world_frame",
    )
    assert review["completed_cache_cycles"] == 1


@pytest.mark.parametrize(
    "fault", ["wrong_transaction", "failed_reply", "timeout", "duplicate_begin"]
)
def test_lift_race_failure_latches_and_cannot_be_repaired_by_later_good_reply(ground, fault):
    start(ground)
    with pytest.raises(ValueError):
        if fault == "wrong_transaction":
            ack(ground, "b" * 64)
        elif fault == "failed_reply":
            ack(ground, response=b"data: false\n")
        elif fault == "duplicate_begin":
            start(ground, "b" * 64)
        else:
            ground.replay.apply("backend_probe_snapshot", {"received_monotonic_sec": 100.81})
    assert ground.replay.fault
    with pytest.raises(ValueError, match="remains latched"):
        ack(ground)
    assert ground.replay.tracker.cycles == 0


def test_unbegun_transaction_reply_cannot_qualify_ground(ground):
    with pytest.raises(ValueError, match="lacks original transaction begin"):
        ack(ground)


def test_lift_begin_requires_measured_ground_not_just_declared_scene(module, tmp_path):
    tape = Tape(module, tmp_path / "source.jsonl")
    with pytest.raises(ValueError, match="fresh measured ground"):
        start(tape)


def test_closed_tape_cannot_hide_unfinished_lift_behind_previous_success(ground):
    ground.cycle()
    start(ground)
    ground.write()
    with pytest.raises(ValueError, match="no complete cache intervention"):
        ground.module.closed_probe_replay(
            ground.path,
            ground.policy,
            plugin_path=ground.path.parent / "unused",
            pose_frame="actual_world_frame",
        )


def test_begin_stale_ground_source_is_refused(ground):
    request = importlib.import_module("probe_lift_evidence").lift_request(ground.policy)
    with pytest.raises(ValueError, match="fresh measured ground"):
        ground.replay.apply(
            "backend_probe_lift_begin",
            {
                "transaction_id": "c" * 64,
                "request": retained(request, "gz.msgs.Pose_protobuf_text"),
                "received_monotonic_sec": 100.71,
                "received_unix_ns": ground.unix + 710_000_000,
            },
        )
