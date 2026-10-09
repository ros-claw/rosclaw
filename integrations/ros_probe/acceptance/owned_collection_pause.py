"""Operator-owned negative collection fixture: only one owned observer child.

No ROS/Gazebo transport, robot signal, action permission or stop proof. Actual
Native failure receipts and independent physical standstill remain separate.
"""

import hashlib
import json
import os
import signal
import time
from pathlib import Path

from backend_probe_world import bounded_source
from probe_scene_geometry import decode_scene_json


class OwnedCollectionPause:
    def __init__(self, directory, children, plan, policy_path=None):
        self.directory, self.children, self.plan = Path(directory), children, plan
        self.policy_path = Path(policy_path) if policy_path is not None else None
        self.original = None
        self.policy = None
        self.child = None
        self.resume_at = None
        self.done = False
        if self.policy_path is not None:
            self.original = bounded_source(self.policy_path, 65536)
            if not 0 < len(self.original) <= 65536:
                raise ValueError("bounded registered collection pause policy required")
            policy = decode_scene_json(self.original)
            expected = {
                key: plan[key] for key in ("run_id", "body_snapshot_hash", "constraint_policy_hash")
            }
            if (
                type(policy) is not dict
                or set(policy)
                != {*expected, "schema_version", "source", "approved", "pause_wall_sec"}
                or policy["schema_version"] != "rosclaw.collection_pause_fixture.v1"
                or policy["source"] != "operator_controlled_SIM_collection_fault"
                or policy["approved"] is not True
                or any(policy[k] != v for k, v in expected.items())
                or type(policy["pause_wall_sec"]) is not int
                or not 1 <= policy["pause_wall_sec"] <= 10
            ):
                raise ValueError("exact bounded owned SIM collector policy required")
            self.policy = policy

    def poll(self):
        if (
            self.child is not None
            and self.resume_at is not None
            and time.monotonic() >= self.resume_at
        ):
            self.resume()
        if (
            self.policy_path is not None
            and bounded_source(self.policy_path, 65536) != self.original
        ):
            raise ValueError("registered collection pause policy changed")
        request = self.directory / "backend-collection-pause-request.json"
        if not request.exists() or self.done or self.child is not None:
            return
        if self.policy is None:
            raise ValueError("unregistered collection pause request refused")
        raw = bounded_source(request, 65536)
        if not 0 < len(raw) <= 65536 or decode_scene_json(raw) != {
            "fixture_policy_sha256": hashlib.sha256(self.original).hexdigest()
        }:
            raise ValueError("collection pause request differs from frozen policy")
        matches = [
            child
            for name, child in self.children.children
            if name == "backend-independent-observer"
        ]
        if len(matches) != 1 or matches[0].poll() is not None:
            raise ValueError("one live owned original observer child required")
        self.child = matches[0]
        record = {
            "schema_version": "rosclaw.owned_collection_pause_record.v1",
            "observer_pid": self.child.pid,
            "fixture_policy": self.policy,
            "fixture_policy_sha256": hashlib.sha256(self.original).hexdigest(),
            "request_original_utf8": raw.decode("utf-8"),
            "request_sha256": hashlib.sha256(raw).hexdigest(),
            "requested_monotonic_sec": time.monotonic(),
            "World_or_robot_signaled": False,
            "physical_stop_proof": "NOT_MEASURED",
            "authorization": False,
        }
        with (self.directory / "backend-collection-pause-record.json").open("x") as output:
            json.dump(record, output, indent=2)
        os.killpg(self.child.pid, signal.SIGSTOP)
        self.resume_at = time.monotonic() + self.policy["pause_wall_sec"]

    def resume(self):
        if self.child is None or self.done:
            return
        if self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGCONT)
        with (self.directory / "backend-collection-resume-record.json").open("x") as output:
            json.dump(
                {
                    "observer_pid": self.child.pid,
                    "resumed_monotonic_sec": time.monotonic(),
                    "collector_exit_code": self.child.poll(),
                    "healthy_source_restored": False,
                    "physical_stop_proof": "NOT_MEASURED",
                },
                output,
                indent=2,
            )
        self.done = True
