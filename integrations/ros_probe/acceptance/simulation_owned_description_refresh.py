"""SDK operator refresh of the identical, original RSP robot_description.

An entity-replacement fixture may need the existing RSP to repeat its latched
description for the recreated controller manager. No new publisher or TF is
created, and the immutable URDF bytes must match before and after the service.
"""

import hashlib
import json
import os
import stat
import time

from simulation_current_facts import _read, read_current_native_pointer


class OwnedDescriptionRefresh:
    def __init__(self, root, assets, transport):
        self.root, self.transport = root, transport
        self.urdf = _read(assets / "robot.urdf", 5_000_000)
        self.sha256 = hashlib.sha256(self.urdf).hexdigest()
        pointer, _ = read_current_native_pointer(root)
        raw = _read(root / pointer["file"], 10_000_000)
        if hashlib.sha256(raw).hexdigest() != pointer["sha256"]:
            raise ValueError("original Native provenance required for description refresh")
        native = json.loads(raw)
        rows = [
            (node, row)
            for node, row in native["observations"]["urdf_descriptions"].items()
            if row.get("sha256") == self.sha256
            and row.get("complete") is True
            and row.get("source") == node + "/get_parameters"
        ]
        if len(rows) != 1:
            raise ValueError("one explicitly observed original URDF parameter node required")
        self.node = rows[0][0]
        topics = [
            row
            for row in native["graph"]["topics"]
            if row.get("msg_type") == "std_msgs/msg/String" and row.get("publishers") == [self.node]
        ]
        if len(topics) != 1:
            raise ValueError("unique original description publisher/topic required")
        self.topic = topics[0]["name"]
        for suffix, kind in [
            ("get_parameters", "GetParameters"),
            ("set_parameters", "SetParameters"),
        ]:
            rows = [
                r for r in native["graph"]["services"] if r.get("name") == self.node + "/" + suffix
            ]
            if rows != [
                {"name": self.node + "/" + suffix, "srv_type": "rcl_interfaces/srv/" + kind}
            ]:
                raise ValueError("unique typed original RSP parameter services required")
        self.exchanges = []
        self._verify_description()

    def _call(self, suffix, kind, request):
        before = time.monotonic_ns()
        response = self.transport.call_service(
            self.node + "/" + suffix,
            request,
            service_type="rcl_interfaces/srv/" + kind,
            timeout_sec=1,
        )
        self.exchanges.append(
            {
                "service": self.node + "/" + suffix,
                "request": request,
                "response_ok": response.ok,
                "original_exchange": response.data,
                "original_error": response.error,
                "before_monotonic_ns": before,
                "after_monotonic_ns": time.monotonic_ns(),
            }
        )
        if response.ok is not True:
            raise ValueError("actual identical-description parameter exchange failed")
        return (response.data or {}).get("values", {})

    def _verify_description(self):
        values = self._call(
            "get_parameters", "GetParameters", {"names": ["robot_description"]}
        ).get("values", [])
        if (
            len(values) != 1
            or values[0].get("type") != 4
            or values[0].get("string_value", "").encode() != self.urdf
        ):
            raise ValueError("current robot_description differs from immutable original URDF")

    def after_create(self, log_offset):
        deadline = time.monotonic_ns() + 2_000_000_000
        needle = ("Subscribing to '" + self.topic + "' topic for robot description.").encode()
        while time.monotonic_ns() < deadline:
            fd = os.open(self.root / "gazebo.log", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or not log_offset <= info.st_size <= 10_000_000:
                    raise ValueError("bounded original Gazebo log required")
                raw = os.pread(fd, min(info.st_size - log_offset, 128_000), log_offset)
            finally:
                os.close(fd)
            if needle in raw:
                break
            time.sleep(0.02)
        else:
            raise ValueError("actual recreated controller description subscriber not observed")
        self._verify_description()
        values = self._call(
            "set_parameters",
            "SetParameters",
            {
                "parameters": [
                    {
                        "name": "robot_description",
                        "value": {"type": 4, "string_value": self.urdf.decode()},
                    }
                ]
            },
        )
        results = values.get("results", [])
        if len(results) != 1 or results[0].get("successful") is not True:
            raise ValueError("identical original description refresh not acknowledged")
        self._verify_description()
        return {
            "node": self.node,
            "topic": self.topic,
            "urdf_sha256": self.sha256,
            "changed_URDF_bytes": False,
            "new_publishers_created": 0,
            "new_TF_publishers_created": 0,
            "subscriber_log_offset": log_offset,
            "subscriber_log_bytes": len(raw),
            "subscriber_log_sha256": hashlib.sha256(raw).hexdigest(),
            "original_exchanges": list(self.exchanges),
            "after_monotonic_ns": time.monotonic_ns(),
        }
