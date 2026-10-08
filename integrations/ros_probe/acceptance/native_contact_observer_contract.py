"""Exercise the real passive process callbacks with official ROS message types.

The Node, clock and delivery loop are synthetic. No DDS, simulation, publisher,
service or motion is started. This verifies recording/closure/replay wiring.
"""

import argparse
import json
import signal
import sys
import tempfile
import time
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path

import native_contact_observer as contact_observer
import rclpy
import rclpy.node
from closed_native_contact_evidence import closed_native_contact_window as closed_contact_window
from geometry_msgs.msg import TransformStamped
from native_contact_cdr_contract import fixture
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

import rosclaw.connectors.ros.diagnosis.coverage_audit as audit_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        help="Exclusive owned directory retaining all synthetic source cases",
    )
    parser.add_argument("--plugin", required=True, type=Path)
    args = parser.parse_args()
    if args.directory is not None:
        args.directory.mkdir(parents=True, exist_ok=False)
    real_pause = time.sleep
    clock = [0.0]
    origin = datetime(2026, 10, 8, 12, tzinfo=UTC)
    nodes = []
    current_case = ["valid"]
    template = [None]
    state = {"init": 0, "shutdown": 0, "destroy": 0}

    class FakeDatetime:
        @staticmethod
        def now(tz):
            return origin + timedelta(seconds=clock[0])

    class FakeNode:
        def __init__(self, name, **options):
            assert options["enable_rosout"] is False
            assert options["start_parameter_services"] is False
            assert options["enable_logger_service"] is False
            assert options["use_global_arguments"] is False
            assert options["parameter_overrides"][0].value is False
            self.publishers = []
            self.services = []
            self.subscriptions = {}
            nodes.append(self)

        def create_subscription(self, kind, topic, callback, qos):
            self.subscriptions[topic] = callback

        def create_publisher(self, *args, **kwargs):
            raise AssertionError("passive process attempted a publisher")

        def create_service(self, *args, **kwargs):
            raise AssertionError("passive process attempted a service")

        def destroy_node(self):
            state["destroy"] += 1

    def spin(node, timeout_sec):
        import copy

        clock[0] = round(clock[0] + 0.1, 6)
        transform = TransformStamped()
        source_time = (
            clock[0] - 0.5 if current_case[0] == "pose_regression" and clock[0] == 2 else clock[0]
        )
        transform.header.stamp.sec, transform.header.stamp.nanosec = divmod(
            round(source_time * 1e9), 1_000_000_000
        )
        transform.header.frame_id = (
            "foreign" if current_case[0] == "frame" and clock[0] == 2 else "fixture_world"
        )
        transform.child_frame_id = "anonymous_body"
        transform.transform.rotation.w = 1.0
        node.subscriptions["/actual_pose"](TFMessage(transforms=[transform]))
        if current_case[0] == "source_pause" and clock[0] >= 2:
            real_pause(0.002)
            return
        packet = copy.deepcopy(template[0])
        packet.update(
            sequence=round(clock[0] * 10),
            iterations=round(clock[0] * 1000),
            sim_time_sec=clock[0],
            captured_at_unix_ns=int((origin + timedelta(seconds=clock[0])).timestamp() * 1e9),
        )
        if clock[0] == 2:
            if current_case[0] == "sequence":
                packet["sequence"] = 1
            elif current_case[0] == "support_loss":
                packet["collision_contacts"][0]["contacts"] = []
            elif current_case[0] == "missing_data":
                packet["complete"] = False
            elif current_case[0] == "contact":
                packet["collision_contacts"][0]["contacts"].append(
                    {
                        "collision1_entity_id": 6,
                        "collision2_entity_id": 900,
                        "collision1_name": "anonymous_body::actual_base::actual_collision",
                        "collision2_name": "synthetic_obstacle::link::collision",
                    }
                )
        node.subscriptions["/rosclaw_sim/contact_components"](String(data=json.dumps(packet)))
        real_pause(0.002)

    original = {
        "node": rclpy.node.Node,
        "init": rclpy.init,
        "spin": rclpy.spin_once,
        "shutdown": rclpy.shutdown,
        "monotonic": time.monotonic,
        "time_ns": time.time_ns,
        "audit_datetime": audit_module.datetime,
        "argv": sys.argv[:],
        "sigint": signal.getsignal(signal.SIGINT),
        "sigterm": signal.getsignal(signal.SIGTERM),
    }
    try:
        rclpy.node.Node = FakeNode
        rclpy.init = lambda: state.__setitem__("init", state["init"] + 1)
        rclpy.shutdown = lambda: state.__setitem__("shutdown", state["shutdown"] + 1)
        rclpy.spin_once = spin
        time.monotonic = lambda: 100 + clock[0]
        time.time_ns = lambda: int((origin + timedelta(seconds=clock[0])).timestamp() * 1e9)
        audit_module.datetime = FakeDatetime
        context = (
            nullcontext(str(args.directory))
            if args.directory is not None
            else tempfile.TemporaryDirectory(prefix="independent-contact-callback-contract-")
        )
        with context as directory:
            results = []
            for case in (
                "valid",
                "source_pause",
                "support_loss",
                "contact",
                "sequence",
                "pose_regression",
                "frame",
                "missing_data",
                "audit_collision",
            ):
                current_case[0], clock[0] = case, 0.0
                nodes.clear()
                state.update(init=0, shutdown=0, destroy=0)
                root = Path(directory) / case
                policy, fixture_rows, _ = fixture(root, args.plugin)
                import base64

                template[0] = json.loads(
                    base64.b64decode(fixture_rows[1]["payload"]["original_source_base64"])
                )
                policy_path = root / "contact-policy.json"
                policy_path.write_text(json.dumps(policy))
                sys.argv = [
                    "contact_observer",
                    "--directory",
                    str(root),
                    "--policy",
                    str(policy_path),
                    "--duration",
                    "60",
                    "--plugin",
                    str(root / "libcontacts.so"),
                    "--pose-frame",
                    "fixture_world",
                ]
                if case == "audit_collision":
                    collision = root / "native-contact-events.jsonl"
                    collision.write_bytes(
                        b"synthetic preexisting evidence must not be overwritten\n"
                    )
                    try:
                        contact_observer.main()
                    except FileExistsError:
                        assert state == {"init": 1, "shutdown": 1, "destroy": 1}
                        assert (
                            collision.read_bytes()
                            == b"synthetic preexisting evidence must not be overwritten\n"
                        )
                        results.append(
                            {"case": case, "status": "EXPECTED_REJECTION_WITH_RESOURCE_CLEANUP"}
                        )
                        continue
                    raise AssertionError("observer overwrote existing evidence")
                contact_observer.main()
                assert len(nodes) == 1 and state == {"init": 1, "shutdown": 1, "destroy": 1}
                latest = json.loads((root / "native-contact-latest.json").read_text())
                summary = json.loads(
                    (root / "native-contact-events.jsonl.summary.json").read_text()
                )
                assert (
                    summary["complete"]
                    and summary["writer_stopped"]
                    and summary["dropped_events"] == 0
                )
                try:
                    report = closed_contact_window(
                        root / "native-contact-events.jsonl",
                        policy,
                        plugin_path=root / "libcontacts.so",
                        pose_frame="fixture_world",
                        start_sim=1.0,
                        end_sim=4.0,
                        start_wall=(origin + timedelta(seconds=1)).isoformat(),
                        end_wall=(origin + timedelta(seconds=4)).isoformat(),
                    )
                except ValueError:
                    if case == "valid":
                        raise
                    assert latest["collision_count"] > 0 or not latest["observation_complete"]
                    results.append({"case": case, "status": "EXPECTED_REJECTION_WITH_CLOSED_AUDIT"})
                else:
                    assert case == "valid", "faulted actual callback path was accepted: " + case
                    assert latest["observation_complete"] and latest["collision_count"] == 0
                    results.append(
                        {
                            "case": case,
                            "status": "PASS_OFFLINE_CALLBACK_CDR_REPLAY",
                            "samples": report["sample_count"],
                        }
                    )
            print(
                json.dumps(
                    {
                        "status": "PASS_OFFLINE_CALLBACK_AND_OFFICIAL_CDR_REPLAY",
                        "actual_ros_message_types": True,
                        "actual_dds_or_simulator_started": False,
                        "native_json_frames": "SYNTHETIC_DERIVED_SDK_FIXTURES",
                        "backend_health_admitted": False,
                        "physical_acceptance": "NOT_RUN",
                        "cases": results,
                    },
                    indent=2,
                )
            )
    finally:
        rclpy.node.Node = original["node"]
        rclpy.init, rclpy.shutdown, rclpy.spin_once = (
            original["init"],
            original["shutdown"],
            original["spin"],
        )
        time.monotonic = original["monotonic"]
        time.time_ns = original["time_ns"]
        audit_module.datetime = original["audit_datetime"]
        sys.argv = original["argv"]
        signal.signal(signal.SIGINT, original["sigint"])
        signal.signal(signal.SIGTERM, original["sigterm"])


if __name__ == "__main__":
    main()
