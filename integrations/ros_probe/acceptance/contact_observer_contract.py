"""Exercise the real passive process callbacks with official ROS message types.

The Node, clock and delivery loop are synthetic. No DDS, simulation, publisher,
service or motion is started. This verifies recording/closure/replay wiring.
"""

import json
import signal
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import contact_observer
import rclpy
import rclpy.node
from closed_contact_evidence import closed_contact_window
from contact_cdr_contract import prepared_sources
from geometry_msgs.msg import TransformStamped
from ros_gz_interfaces.msg import Contact, Contacts
from tf2_msgs.msg import TFMessage

import rosclaw.connectors.ros.diagnosis.coverage_audit as audit_module


def main():
    real_pause = time.sleep
    clock = [0.0]
    origin = datetime(2026, 10, 8, 12, tzinfo=UTC)
    nodes = []
    current_case = ["valid"]
    state = {"init": 0, "shutdown": 0, "destroy": 0}

    class FakeDatetime:
        @staticmethod
        def now(tz):
            return origin + timedelta(seconds=clock[0])

    class FakeNode:
        def __init__(self, name):
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
        clock[0] = round(clock[0] + 0.1, 6)
        transform = TransformStamped()
        source_time = (
            clock[0] - 0.5 if current_case[0] == "pose_regression" and clock[0] == 2 else clock[0]
        )
        ns = round(source_time * 1e9)
        transform.header.stamp.sec, transform.header.stamp.nanosec = divmod(ns, 1_000_000_000)
        transform.child_frame_id = "anonymous_body"
        transform.transform.rotation.w = 1.0
        node.subscriptions["/rosclaw_sim/ground_truth"](TFMessage(transforms=[transform]))
        for topic in ("/contacts/support", "/contacts/body"):
            if current_case[0] == "missing_body" and topic == "/contacts/body":
                continue
            if current_case[0] == "support_loss" and topic == "/contacts/support" and clock[0] >= 2:
                continue
            message = Contacts()
            message.header.stamp = transform.header.stamp
            if topic == "/contacts/support":
                contact = Contact()
                contact.collision1.name = "anonymous_body::support::solid"
                contact.collision2.name = "floor::ground::solid"
                message.contacts = [contact]
            elif current_case[0] in {"contact", "foreign_contact"} and clock[0] == 2:
                contact = Contact()
                contact.collision1.name = (
                    "anonymous_body::base::solid"
                    if current_case[0] == "contact"
                    else "foreign::base::solid"
                )
                contact.collision2.name = "obstacle::base::solid"
                message.contacts = [contact]
            if current_case[0] == "invalid_stamp" and clock[0] == 2:
                message.header.stamp.nanosec = 1_000_000_000
            node.subscriptions[topic](message)
        real_pause(0.002)  # Give the real bounded audit writer an ordinary scheduling opportunity.

    original = {
        "node": rclpy.node.Node,
        "init": rclpy.init,
        "spin": rclpy.spin_once,
        "shutdown": rclpy.shutdown,
        "monotonic": time.monotonic,
        "observer_datetime": contact_observer.datetime,
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
        contact_observer.datetime = audit_module.datetime = FakeDatetime
        with tempfile.TemporaryDirectory(
            prefix="independent-contact-callback-contract-"
        ) as directory:
            results = []
            for case in (
                "valid",
                "missing_body",
                "support_loss",
                "contact",
                "foreign_contact",
                "pose_regression",
                "invalid_stamp",
            ):
                current_case[0], clock[0] = case, 0.0
                nodes.clear()
                state.update(init=0, shutdown=0, destroy=0)
                root = Path(directory) / case
                root.mkdir()
                policy = prepared_sources(root)
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
                ]
                contact_observer.main()
                assert len(nodes) == 1 and state == {"init": 1, "shutdown": 1, "destroy": 1}
                latest = json.loads((root / "independent-contact-latest.json").read_text())
                summary = json.loads(
                    (root / "independent-contact-events.jsonl.summary.json").read_text()
                )
                assert (
                    summary["complete"]
                    and summary["writer_stopped"]
                    and summary["dropped_events"] == 0
                )
                try:
                    report = closed_contact_window(
                        root / "independent-contact-events.jsonl",
                        policy,
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
        contact_observer.datetime, audit_module.datetime = (
            original["observer_datetime"],
            original["audit_datetime"],
        )
        sys.argv = original["argv"]
        signal.signal(signal.SIGINT, original["sigint"])
        signal.signal(signal.SIGTERM, original["sigterm"])


if __name__ == "__main__":
    main()
