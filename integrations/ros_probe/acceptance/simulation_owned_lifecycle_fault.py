"""SDK operator fault on an explicitly observed collision-monitor lifecycle.

No Runtime, DDS, publishers or goal client. The owned actor supplies an already
connected isolated rosbridge transport and immutable context; this is no tool.
"""

import hashlib
import json
import time

from simulation_current_facts import _read, read_current_native_pointer


class OwnedCollisionMonitorFault:
    def __init__(self, root, transport, *, run_id, body_snapshot_hash):
        self.root, self.transport = root, transport
        self.run_id, self.body_snapshot_hash = run_id, body_snapshot_hash
        requirements = json.loads(
            _read(root / "generic-compiled/navigation-requirements.json", 32768)
        )
        self.node = requirements["lifecycle_nodes"]["collision_monitor"]
        if type(self.node) is not str or not self.node.startswith("/") or len(self.node) > 512:
            raise ValueError("explicit bounded collision monitor required")
        pointer, _ = read_current_native_pointer(root)
        raw = _read(root / pointer["file"], 10_000_000)
        if hashlib.sha256(raw).hexdigest() != pointer["sha256"]:
            raise ValueError("original Native lifecycle service bytes required")
        services = json.loads(raw)["graph"]["services"]
        for suffix, kind in [("get_state", "GetState"), ("change_state", "ChangeState")]:
            rows = [x for x in services if x.get("name") == self.node + "/" + suffix]
            if rows != [
                {"name": self.node + "/" + suffix, "srv_type": "lifecycle_msgs/srv/" + kind}
            ]:
                raise ValueError("unique original typed collision lifecycle services required")
        self.exchanges = []

    def _call(self, suffix, kind, request):
        started = time.monotonic_ns()
        response = self.transport.call_service(
            self.node + "/" + suffix,
            request,
            service_type="lifecycle_msgs/srv/" + kind,
            timeout_sec=1,
        )
        self.exchanges.append(
            {
                "service": self.node + "/" + suffix,
                "request": request,
                "response_ok": response.ok,
                "original_exchange": response.data,
                "original_error": response.error,
                "before_monotonic_ns": started,
                "after_monotonic_ns": time.monotonic_ns(),
            }
        )
        if response.ok is not True:
            raise ValueError("actual collision lifecycle exchange failed: " + str(response.error))
        return (response.data or {}).get("values", {})

    def _state(self, expected):
        state = self._call("get_state", "GetState", {}).get("current_state", {})
        if type(state.get("id")) is not int or state["id"] != expected:
            raise ValueError("actual collision monitor lifecycle state differs")

    def _transition(self, transition):
        if (
            self._call("change_state", "ChangeState", {"transition": {"id": transition}}).get(
                "success"
            )
            is not True
        ):
            raise ValueError("collision lifecycle transition was not acknowledged")

    def disable(self):
        self._state(3)
        self._transition(4)
        self._state(2)
        return {"node": self.node, "actual_state_id": 2, "original_exchanges": list(self.exchanges)}

    def restore_after_terminal_stop(self):
        sink = json.loads(
            _read(self.root / "governed-sink-status.json", 4096, allow_atomic_unlink=True)
        )
        if (
            sink.get("terminal") is not True
            or sink.get("lease_expires_ns") != 0
            or sink.get("run_id") != self.run_id
            or sink.get("body_snapshot_hash") != self.body_snapshot_hash
        ):
            raise ValueError("same-context terminal lease-closed sink required before restoration")
        self._state(2)
        self._transition(3)
        self._state(3)
        return {
            "node": self.node,
            "actual_state_id": 3,
            "restored_after_terminal_sink": sink,
            "original_exchanges": list(self.exchanges),
            "after_monotonic_ns": time.monotonic_ns(),
        }
