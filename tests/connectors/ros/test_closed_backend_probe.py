"""Synthetic original-tape integrity tests; pose decoder is a declared test double.

Actual installed ROS CDR decoding and actual gravity episodes remain separate.
"""

import base64
import copy
import hashlib
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from tests.connectors.ros.test_backend_probe_evidence import specification
from tests.connectors.ros.test_native_contact_components import packet_named


def retained(raw, kind):
    return {
        "original_source_base64": base64.b64encode(raw).decode(),
        "original_source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_size_bytes": len(raw),
        "source_bytes_complete": True,
        "source_type": kind,
    }


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("closed_backend_probe")

    def synthetic_pose(raw, model, frame):
        packet = json.loads(raw)
        if packet["model"] != model or packet["frame"] != frame:
            raise ValueError("synthetic pose scope differs")
        return packet["sim"], packet["pose"]

    monkeypatch.setattr(module, "decode_ros_pose", synthetic_pose)
    monkeypatch.setattr(module, "reopen_native_policy", lambda *args, **kwargs: None)
    return module


class Tape:
    def __init__(self, module, path):
        self.module, self.path = module, path
        self.policy = specification()
        self.replay = module.ProbeEventReplay(self.policy, "actual_world_frame")
        self.rows, self.i = [], 0
        self.packet = packet_named("native_actual_contact_entity_names")
        self.unix = 1_791_504_000_000_000_000
        base = self.policy["native_policy"]["contact_policy"]
        self.context = {
            **{
                k: base[k]
                for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
            },
            "source": "owned_SIM_backend_probe_original_sources",
            "evidence_domain": "SIMULATION",
            "probe_policy_hash": digest(self.policy),
            "pose_frame": "actual_world_frame",
        }

    def emit(self, kind, payload):
        payload["projection"] = self.replay.apply(kind, payload)
        row = {
            "schema_version": "rosclaw.coverage_audit_event.v1",
            **self.context,
            "kind": kind,
            "payload": payload,
            "wall_monotonic_sec": payload["received_monotonic_sec"] + 0.0001,
            "sim_time_sec": payload["projection"].get("sim_time_sec")
            if kind in {"backend_probe_pose", "backend_probe_components"}
            else self.replay.tracker.previous[2]
            if kind == "backend_probe_snapshot" and self.replay.tracker.previous
            else None,
            "captured_at": datetime.fromtimestamp(
                (self.unix + round((payload["received_monotonic_sec"] - 100) * 1e9) + 100_000)
                / 1e9,
                UTC,
            ).isoformat(),
        }
        self.rows.append(row)

    def frame(self, z=0.05, touching=True):
        self.i += 1
        sim, wall = self.i * 0.05, 100 + self.i * 0.05
        xyz = [6, 0, z, 1, 0, 0, 0]
        raw_pose = json.dumps(
            {
                "model": self.packet["body_model_name"],
                "frame": "actual_world_frame",
                "sim": sim,
                "pose": xyz,
            }
        ).encode()
        self.emit(
            "backend_probe_pose",
            {**retained(raw_pose, "tf2_msgs/msg/TFMessage"), "received_monotonic_sec": wall},
        )
        packet = copy.deepcopy(self.packet)
        packet.update(
            sequence=self.i,
            iterations=self.i * 50,
            sim_time_sec=sim,
            captured_at_unix_ns=self.unix + round(sim * 1e9),
            body_world_pose=xyz,
        )
        if not touching:
            packet["collision_contacts"][0]["contacts"] = []
        self.emit(
            "backend_probe_components",
            {
                **retained(json.dumps(packet).encode(), "gazebo_ecm_contact_sensor_data_json"),
                "received_monotonic_sec": wall + 0.001,
                "received_unix_ns": packet["captured_at_unix_ns"] + 1_000_000,
            },
        )
        self.emit("backend_probe_snapshot", {"received_monotonic_sec": wall + 0.002})

    def cycle(self):
        for _ in range(12):
            self.frame()
        lift_module = importlib.import_module("probe_lift_evidence")
        self.emit(
            "backend_probe_lift_ack",
            {
                "request": retained(
                    lift_module.lift_request(self.policy), "gz.msgs.Pose_protobuf_text"
                ),
                "response": retained(b"data: true\n", "gz.msgs.Boolean_protobuf_text"),
                "returncode": 0,
                "received_monotonic_sec": 100 + self.i * 0.05 + 0.003,
                "received_unix_ns": self.unix + round(self.i * 0.05 * 1e9) + 3_000_000,
            },
        )
        for i in range(12):
            self.frame(z=10 - i * 0.05, touching=False)
        for _ in range(12):
            self.frame()
        self.write()

    def write(self):
        previous = None
        for i, row in enumerate(self.rows, 1):
            row.pop("artifact_sha256", None)
            row.update(sequence=i, previous_hash=previous)
            row["artifact_sha256"] = digest(row)
            previous = row["artifact_sha256"]
        self.path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        Path(str(self.path) + ".summary.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "writer_stopped": True,
                    "dropped_events": 0,
                    "writer_error": None,
                    "events_written": len(self.rows),
                    "last_event_hash": previous,
                }
            )
        )

    def check(self):
        return self.module.closed_probe_replay(
            self.path,
            self.policy,
            plugin_path="unused_synthetic_plugin",
            pose_frame="actual_world_frame",
        )


def test_closed_original_cycle_replays_but_cannot_admit_physics_or_actions(module, tmp_path):
    tape = Tape(module, tmp_path / "probe.jsonl")
    tape.cycle()
    result = tape.check()
    assert result["completed_cache_cycles"] == 1 and result["sample_count"] == 36
    assert result["backend_health_admitted"] is False and result["authorization"] is False
    assert result["physical_acceptance"] == "NOT_VERIFIED"


@pytest.mark.parametrize(
    "fault",
    [
        "original_sha",
        "projection",
        "binding",
        "ack_target",
        "reply_false",
        "regressed_wall",
        "unknown_event",
        "unfinished_line",
        "summary_unclosed",
    ],
)
def test_rehashed_false_source_or_projection_and_unclosed_tapes_fail(module, tmp_path, fault):
    tape = Tape(module, tmp_path / "probe.jsonl")
    tape.cycle()
    component = tape.rows[1]
    ack = next(row for row in tape.rows if row["kind"] == "backend_probe_lift_ack")
    if fault == "original_sha":
        component["payload"]["original_source_sha256"] = "0" * 64
    elif fault == "projection":
        component["payload"]["projection"]["completed_cache_cycles"] = 99
    elif fault == "binding":
        component["run_id"] = "wrong_run"
    elif fault == "ack_target":
        ack["payload"]["request"] = retained(b'name: "actual_robot"', "gz.msgs.Pose_protobuf_text")
    elif fault == "reply_false":
        ack["payload"]["response"] = retained(b"data: false", "gz.msgs.Boolean_protobuf_text")
    elif fault == "regressed_wall":
        component["payload"]["received_monotonic_sec"] = 1
    elif fault == "unknown_event":
        component["kind"] = "claimed_success"
    tape.write()
    if fault == "unfinished_line":
        tape.path.write_bytes(tape.path.read_bytes()[:-1])
    elif fault == "summary_unclosed":
        summary_path = Path(str(tape.path) + ".summary.json")
        summary = json.loads(summary_path.read_text())
        summary["writer_stopped"] = False
        summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        tape.check()


def test_ground_only_has_no_closed_intervention_proof(module, tmp_path):
    tape = Tape(module, tmp_path / "probe.jsonl")
    for _ in range(12):
        tape.frame()
    tape.write()
    with pytest.raises(ValueError, match="no complete"):
        tape.check()


def test_components_arriving_before_exact_pose_are_retained_and_paired_without_interpolation(
    module,
):
    policy = specification()
    replay = module.ProbeEventReplay(policy, "actual_world_frame")
    packet = packet_named("native_actual_contact_entity_names")
    packet.update(
        sequence=1,
        iterations=50,
        sim_time_sec=0.05,
        captured_at_unix_ns=1_791_504_000_050_000_000,
        body_world_pose=[6, 0, 0.05, 1, 0, 0, 0],
    )
    result = replay.apply(
        "backend_probe_components",
        {
            **retained(json.dumps(packet).encode(), "gazebo_ecm_contact_sensor_data_json"),
            "received_monotonic_sec": 100.051,
            "received_unix_ns": packet["captured_at_unix_ns"] + 1_000_000,
        },
    )
    assert result["pending_exact_pose"] == 1 and result["paired_components"] == []
    raw = json.dumps(
        {
            "model": packet["body_model_name"],
            "frame": "actual_world_frame",
            "sim": 0.05,
            "pose": packet["body_world_pose"],
        }
    ).encode()
    result = replay.apply(
        "backend_probe_pose",
        {**retained(raw, "tf2_msgs/msg/TFMessage"), "received_monotonic_sec": 100.06},
    )
    assert len(result["paired_components"]) == 1 and not replay.pending
    assert replay.tracker.previous[2] == 0.05
    assert not replay.tracker.snapshot(100.061)["cache_update_pattern_observed"]


def test_unpaired_original_component_expires_and_rejection_cannot_be_skipped(module):
    replay = module.ProbeEventReplay(specification(), "actual_world_frame")
    packet = packet_named("native_actual_contact_entity_names")
    replay.apply(
        "backend_probe_components",
        {
            **retained(json.dumps(packet).encode(), "gazebo_ecm_contact_sensor_data_json"),
            "received_monotonic_sec": 100,
            "received_unix_ns": packet["captured_at_unix_ns"] + 1_000_000,
        },
    )
    with pytest.raises(ValueError, match="timely exact pose"):
        replay.apply("backend_probe_snapshot", {"received_monotonic_sec": 100.31})
    with pytest.raises(ValueError, match="latched"):
        replay.apply("backend_probe_snapshot", {"received_monotonic_sec": 100.32})
