"""Replay a closed instrument source tape; never promote runtime admission.

Original TF CDR, component JSON and service request/reply bytes are replayed.
Hash-chain integrity proves correspondence, not publisher authentication.
"""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from backend_probe_evidence import BackendCacheProbe, probe_policy
from closed_native_contact_evidence import decode_ros_pose, original_ros_bytes
from native_contact_evidence import decode_native_packet, reopen_native_policy
from probe_lift_evidence import derive_lift_ack, lift_request

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


class ProbeEventReplay:
    """Apply retained original events identically online and after closure."""

    def __init__(self, policy, pose_frame):
        probe_policy(policy)
        if type(pose_frame) is not str or not 0 < len(pose_frame) <= 256:
            raise ValueError("explicit actual instrument world frame required")
        self.policy, self.pose_frame = policy, pose_frame
        self.tracker = BackendCacheProbe(policy)
        self.samples = []
        self.last_wall = None
        self.fault = None
        self.pending = []
        self.lift_transaction = None
        self.used_lift_transactions = set()

    def transaction_fresh(self, wall):
        if self.lift_transaction is not None and wall - self.lift_transaction[1] >= 0.2:
            self.fault = (
                self.fault or "original instrument lift transaction acknowledgement timed out"
            )
        if self.fault:
            raise ValueError(self.fault)

    def apply(self, kind, payload):
        if self.fault:
            raise ValueError("instrument source rejection remains latched")
        try:
            return self._apply(kind, payload)
        except (ValueError, KeyError, TypeError) as exc:
            self.fault = str(exc)
            raise ValueError(f"original instrument source rejected: {exc}") from exc

    def _apply(self, kind, payload):
        wall = payload.get("received_monotonic_sec")
        if (
            type(wall) not in (int, float)
            or not math.isfinite(wall)
            or wall < 0
            or (self.last_wall is not None and wall < self.last_wall)
        ):
            raise ValueError("original instrument event receipt clock regressed")
        self.last_wall = wall
        self.transaction_fresh(wall)
        if self.pending and wall - self.pending[0][1] >= 0.3:
            raise ValueError("pending original instrument component lacks timely exact pose")
        if kind == "backend_probe_pose":
            raw = original_ros_bytes(payload, "tf2_msgs/msg/TFMessage")
            sim, pose = decode_ros_pose(
                raw, self.policy["native_policy"]["contact_policy"]["model_name"], self.pose_frame
            )
            self.tracker.pose(sim, wall, pose)
            return {"sim_time_sec": sim, "paired_components": self._drain()}
        if kind == "backend_probe_components":
            raw = original_ros_bytes(payload, "gazebo_ecm_contact_sensor_data_json")
            packet, _ = decode_native_packet(raw, self.policy["native_policy"])
            if len(self.pending) >= 32:
                raise ValueError("original component exact-pose buffer overflow")
            self.pending.append((raw, wall, payload["received_unix_ns"], packet["sim_time_sec"]))
            return {
                "sim_time_sec": packet["sim_time_sec"],
                "paired_components": self._drain(),
                "pending_exact_pose": len(self.pending),
            }
        if kind == "backend_probe_lift_begin":
            request = original_ros_bytes(payload["request"], "gz.msgs.Pose_protobuf_text")
            transaction = payload.get("transaction_id")
            unix = payload.get("received_unix_ns")
            if (
                self.lift_transaction is not None
                or self.pending
                or self.tracker.phase != "READY_FOR_LIFT"
                or self.tracker.previous is None
                or type(unix) is not int
                or not 0 <= unix - self.tracker.previous[3] < 100_000_000
                or type(transaction) is not str
                or len(transaction) != 64
                or any(c not in "0123456789abcdef" for c in transaction)
                or transaction in self.used_lift_transactions
                or request != lift_request(self.policy)
            ):
                raise ValueError(
                    "fresh measured ground and unique exact instrument lift begin required"
                )
            if len(self.used_lift_transactions) >= 500:
                raise ValueError("bounded immutable instrument lift transaction count exceeded")
            self.used_lift_transactions.add(transaction)
            self.lift_transaction = (transaction, wall, unix)
            return {
                "transaction_id": transaction,
                "request_sha256": hashlib.sha256(request).hexdigest(),
                "lift_transaction_pending": True,
            }
        if kind == "backend_probe_lift_ack":
            transaction = self.lift_transaction
            completed = payload.get("acknowledged_at_unix_ns", payload["received_unix_ns"])
            if (
                type(completed) is not int
                or type(payload.get("received_unix_ns")) is not int
                or not 0 <= payload["received_unix_ns"] - completed < 100_000_000
            ):
                raise ValueError("original instrument service completion receipt stale or future")
            if transaction is not None and (
                payload.get("transaction_id") != transaction[0]
                or not 0 <= completed - transaction[2] < 200_000_000
            ):
                raise ValueError("original instrument lift ACK does not match pending transaction")
            if transaction is None and payload.get("transaction_id") is not None:
                raise ValueError("instrument lift ACK lacks original transaction begin")
            if self.pending and transaction is None:
                raise ValueError("instrument lift cannot bypass an unpaired original source")
            request = original_ros_bytes(payload["request"], "gz.msgs.Pose_protobuf_text")
            response = original_ros_bytes(payload["response"], "gz.msgs.Boolean_protobuf_text")
            ack = derive_lift_ack(
                self.policy,
                request_bytes=request,
                response_bytes=response,
                returncode=payload["returncode"],
                acknowledged_at_unix_ns=completed,
            )
            self.tracker.acknowledge_lift(ack)
            self.lift_transaction = None
            paired = self._drain()
            return {
                **ack,
                **(
                    {"paired_components": paired, "transaction_id": transaction[0]}
                    if transaction is not None
                    else {}
                ),
            }
        if kind == "backend_probe_snapshot":
            result = self.tracker.snapshot(wall)
            self.samples.append(
                (wall, self.tracker.previous[2] if self.tracker.previous else None, result)
            )
            if len(self.samples) > 50000:
                raise ValueError("bounded immutable instrument duration exceeded")
            return result
        raise ValueError("unknown/rejected instrument source event")

    def _drain(self):
        results = []
        if self.lift_transaction is not None:
            return results
        while self.pending:
            raw, wall, unix, sim = self.pending[0]
            if round(sim * 1e9) not in self.tracker.pose_history:
                break
            results.append(
                self.tracker.observe(raw, received_monotonic_sec=wall, received_unix_ns=unix)
            )
            self.pending.pop(0)
        return results


def closed_probe_replay(path, policy, *, plugin_path, pose_frame):
    """Check closed source correspondence; actual runtime admission stays pending."""
    replay = ProbeEventReplay(policy, pose_frame)
    path = Path(path)
    reopen_native_policy(path.parent, policy["native_policy"], plugin_path=plugin_path)
    summary_path = Path(str(path) + ".summary.json")
    if (
        path.is_symlink()
        or summary_path.is_symlink()
        or not path.is_file()
        or not summary_path.is_file()
        or not 0 < path.stat().st_size <= 1_000_000_000
        or not 0 < summary_path.stat().st_size <= 65536
    ):
        raise ValueError("bounded regular original closed instrument tape required")
    identity = (path.stat().st_dev, path.stat().st_ino, path.stat().st_size)
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    if (
        summary.get("complete") is not True
        or summary.get("writer_stopped") is not True
        or type(summary.get("dropped_events")) is not int
        or summary["dropped_events"] != 0
        or summary.get("writer_error") is not None
    ):
        raise ValueError("complete stopped lossless instrument writer required")
    previous, sequence, source_hash = None, 0, hashlib.sha256()
    last_capture = None
    base = policy["native_policy"]["contact_policy"]
    with path.open("rb") as stream:
        for line in iter(lambda: stream.readline(2_000_001), b""):
            if len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise ValueError("bounded complete instrument tape row required")
            source_hash.update(line)
            row = json.loads(line)
            saved = row.pop("artifact_sha256", None)
            sequence += 1
            if (
                sequence > 2_000_000
                or row.get("schema_version") != "rosclaw.coverage_audit_event.v1"
                or type(row.get("sequence")) is not int
                or row.get("sequence") != sequence
                or row.get("previous_hash") != previous
                or digest(row) != saved
            ):
                raise ValueError("instrument source genesis/sequence/hash differs")
            previous = saved
            if (
                any(
                    row.get(k) != base[k]
                    for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
                )
                or row.get("source") != "owned_SIM_backend_probe_original_sources"
                or row.get("evidence_domain") != "SIMULATION"
                or row.get("probe_policy_hash") != digest(policy)
                or row.get("pose_frame") != pose_frame
            ):
                raise ValueError("instrument original source binding differs")
            payload = row["payload"]
            captured = datetime.fromisoformat(row["captured_at"])
            if captured.tzinfo is None or (last_capture is not None and captured < last_capture):
                raise ValueError("original instrument capture clock regressed")
            last_capture = captured
            if row["kind"] in {
                "backend_probe_components",
                "backend_probe_lift_ack",
                "backend_probe_lift_begin",
            }:
                unix = payload.get("received_unix_ns")
                if (
                    type(unix) is not int
                    or not 0 <= int(captured.timestamp() * 1e9) - unix < 300_000_000
                ):
                    raise ValueError("instrument original UNIX receipt differs from tape")
            received = payload["received_monotonic_sec"]
            if (
                type(received) not in (int, float)
                or not 0 <= row["wall_monotonic_sec"] - received < 0.3
            ):
                raise ValueError("retained original instrument receipt differs from tape")
            actual = replay.apply(row["kind"], payload)
            if payload.get("projection") != actual:
                raise ValueError(
                    "instrument retained projection differs from original source replay"
                )
            expected_sim = (
                actual["sim_time_sec"]
                if row["kind"] in {"backend_probe_pose", "backend_probe_components"}
                else replay.tracker.previous[2]
                if row["kind"] == "backend_probe_snapshot" and replay.tracker.previous
                else None
            )
            if row.get("sim_time_sec") != expected_sim:
                raise ValueError("instrument tape SIM clock differs from original source")
    if (
        (path.stat().st_dev, path.stat().st_ino, path.stat().st_size) != identity
        or summary_path.read_bytes() != summary_raw
        or type(summary.get("events_written")) is not int
        or summary["events_written"] != sequence
        or summary.get("last_event_hash") != previous
    ):
        raise ValueError("original instrument writer closure/source changed")
    reopen_native_policy(path.parent, policy["native_policy"], plugin_path=plugin_path)
    if (
        replay.fault
        or replay.pending
        or replay.lift_transaction is not None
        or replay.tracker.fault
        or not replay.samples
        or replay.tracker.cycles < 1
    ):
        raise ValueError("closed original instrument tape has no complete cache intervention cycle")
    if not replay.samples[-1][2]["cache_update_pattern_observed"]:
        raise ValueError("instrument pattern not fresh at closed final observation")
    return {
        "evidence_role": "closed_probe_source_correspondence_not_runtime_admission",
        "source_file_sha256": source_hash.hexdigest(),
        "summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "probe_policy_hash": digest(policy),
        "completed_cache_cycles": replay.tracker.cycles,
        "sample_count": len(replay.samples),
        "backend_health_admitted": False,
        "physical_acceptance": "NOT_VERIFIED",
        "authorization": False,
    }
