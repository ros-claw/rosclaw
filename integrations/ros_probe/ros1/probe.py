"""Read-only ROS1 companion; ROS imports stay outside the ROS-free core.

Reads the master graph, published messages and parameters. Its only publishers
are diagnostic snapshot/status topics; it never invokes robot services.
"""

import argparse
import json
import os
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import rosgraph
import roslib.message
import rospy
from std_msgs.msg import String


def utc_now():
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - ROS1 Noetic Python 3.8


class ReadOnlyProbe:
    def __init__(self):
        self.master = rosgraph.Master(rospy.get_name())
        self.lock = threading.RLock()
        self.received = {}
        self.transforms = {}
        self.subscriptions = {}
        self.latched = set()
        self.errors = []
        self.publisher = rospy.Publisher(
            "/rosclaw_probe/snapshot", String, queue_size=1, latch=True
        )
        self.status = rospy.Publisher("/rosclaw_probe/status", String, queue_size=1, latch=True)

    def observe(self, topic, message):
        now = time.monotonic()
        with self.lock:
            self.received.setdefault(topic, deque(maxlen=100)).append(now)
            if getattr(message, "_connection_header", {}).get("latching") == "1":
                self.latched.add(topic)
            if hasattr(message, "transforms"):
                for edge in message.transforms:
                    self.transforms[(edge.header.frame_id, edge.child_frame_id)] = (
                        edge.header.stamp.to_sec(),
                        topic.endswith("tf_static"),
                    )

    def snapshot(self):
        stamp = utc_now()
        publishers, subscribers, services = self.master.getSystemState()
        # Goal topics may have subscribers but no active client publisher.
        types = dict(self.master.getTopicTypes())
        pub_map, sub_map = dict(publishers), dict(subscribers)
        for topic, msg_type in types.items():
            if topic.startswith("/rosclaw_probe/") or topic in self.subscriptions:
                continue
            if msg_type not in {
                "sensor_msgs/LaserScan",
                "sensor_msgs/Image",
                "nav_msgs/Odometry",
                "nav_msgs/OccupancyGrid",
                "geometry_msgs/PoseWithCovarianceStamped",
                "tf2_msgs/TFMessage",
                "rosgraph_msgs/Clock",
            }:
                continue
            cls = roslib.message.get_message_class(msg_type)
            if cls is None:
                self.errors.append(f"message type unavailable: {msg_type}")
                continue
            self.subscriptions[topic] = rospy.Subscriber(
                topic, cls, lambda msg, t=topic: self.observe(t, msg), queue_size=10
            )
        nodes = sorted(
            {n for group in (publishers, subscribers, services) for _, ns in group for n in ns}
        )
        actions = []
        for topic, msg_type in types.items():
            if topic.endswith("/goal") and msg_type.endswith("ActionGoal"):
                name = topic[:-5]
                if (
                    types.get(name + "/status") == "actionlib_msgs/GoalStatusArray"
                    and types.get(name + "/result") == msg_type[:-4] + "Result"
                ):
                    actions.append(
                        {
                            "name": name,
                            "action_type": msg_type[:-4],
                            "servers": pub_map.get(name + "/result", []),
                        }
                    )
        with self.lock:
            signals = []
            for topic in self.subscriptions:
                samples = list(self.received.get(topic, ()))
                intervals = [b - a for a, b in zip(samples, samples[1:])]  # noqa: B905 - Python 3.8
                signal = {
                    "source": "ros1_native_subscription",
                    "captured_at": stamp,
                    "topic": topic,
                    "rate_hz": (len(samples) - 1) / (samples[-1] - samples[0])
                    if len(samples) > 1 and samples[-1] > samples[0]
                    else None,
                    "jitter_ms": statistics.pstdev(intervals) * 1000 if intervals else None,
                    "last_message_age_ms": (time.monotonic() - samples[-1]) * 1000
                    if samples
                    else None,
                    "publisher_count": len(pub_map.get(topic, [])),
                    "subscriber_count": len(sub_map.get(topic, [])),
                }
                if topic in self.latched:
                    signal["freshness_policy"] = "latched"
                signals.append(signal)
            transforms = [
                {
                    "source": "ros1_native_tf_subscription",
                    "captured_at": stamp,
                    "parent": parent,
                    "child": child,
                    "static": static,
                    "age_ms": None if static else (rospy.Time.now().to_sec() - observed) * 1000,
                }
                for (parent, child), (observed, static) in self.transforms.items()
            ]
        return {
            "schema_version": "rosclaw.ros_probe.v1",
            "captured_at": stamp,
            "environment": {
                "ros_generation": "ros1",
                "distro": os.getenv("ROS_DISTRO", "unknown"),
                "use_sim_time": rospy.get_param("/use_sim_time", False),
                "overlays": os.getenv("ROS_PACKAGE_PATH", "").split(":"),
            },
            "graph": {
                "topics": [
                    {
                        "name": n,
                        "msg_type": t,
                        "publishers": pub_map.get(n, []),
                        "subscribers": sub_map.get(n, []),
                    }
                    for n, t in sorted(types.items())
                ],
                "nodes": [{"name": n} for n in nodes],
                "actions": actions,
                "services": [{"name": n, "providers": ns} for n, ns in services],
            },
            "signals": signals,
            "transforms": transforms,
            "lifecycle": [],
            "qos": {"supported": False, "endpoints": []},
            "navigation": {},
            "observations": {"lifecycle_supported": False},
            "completeness": {
                "graph": True,
                "signals": bool(signals),
                "tf": bool(transforms),
                "qos": False,
                "lifecycle": False,
                "time": True,
            },
            "errors": self.errors[-50:],
        }

    def close(self):
        for subscription in self.subscriptions.values():
            subscription.unregister()
        self.publisher.unregister()
        self.status.unregister()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("rosclaw_readonly_probe")
    probe = ReadOnlyProbe()
    try:
        deadline = time.monotonic() + args.duration
        while not rospy.is_shutdown():
            snapshot = probe.snapshot()
            if args.once and time.monotonic() >= deadline:
                result = json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
                if args.output:
                    args.output.write_text(result)
                else:
                    print(result)
                break
            probe.publisher.publish(String(data=json.dumps(snapshot, allow_nan=False)))
            probe.status.publish(
                String(data=json.dumps({"read_only": True, "captured_at": utc_now()}))
            )
            time.sleep(0.2 if args.once else 1)
    finally:
        probe.close()


if __name__ == "__main__":
    main()
