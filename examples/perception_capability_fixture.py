"""Explicit local FIXTURE run of the perception-quality App via a real private daemon.

Installs ``examples/apps/perception-quality`` by explicit local path through
the existing ``AppStore``, then drives the actual private Unix
``DaemonControlPlane``/``DaemonClient``/``AppRunner`` path. Evidence is
FIXTURE/SYNTHETIC (receipts DEGRADED) — this never claims a live sensor or
physical execution, and no Runtime.initialize/robot loading is performed.

Run: ``python -B examples/perception_capability_fixture.py``
"""

import json
import struct
import tempfile
from pathlib import Path

from rosclaw.app.runner import AppRunner
from rosclaw.app.schema import AppManifest
from rosclaw.app.store import AppStore
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.client import DaemonClient
from rosclaw.daemon.ledger import DaemonLedger
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel import ExecutionMode

SCAN_PAYLOAD = {
    "record": {
        "header": {"stamp": {"sec": -1, "nanosec": 0}, "frame_id": "controlled"},
        "angle_min": 0.0,
        "angle_max": 0.25,
        "angle_increment": 0.25,
        "time_increment": 0.0,
        "scan_time": 0.0,
        "range_min": 0.1,
        "range_max": 4.0,
        "ranges": [1.0, 2.0],
        "intensities": [],
    },
    "reference_time_ns": -1_000_000_000,
}

CLOUD_PAYLOAD = {
    "record": {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "controlled"},
        "width": 1,
        "height": 1,
        "point_step": 12,
        "row_step": 12,
        "is_bigendian": False,
        "is_dense": True,
        "fields": [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
        ],
        "data": list(struct.pack("<fff", 0.5, 0.25, 1.0)),
    }
}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="perception-fixture-") as name:
        root = Path(name)
        root.chmod(0o700)
        runtime = Runtime(
            RuntimeConfig(
                robot_id="supplied-fixture",
                enable_firewall=False,
                enable_memory=False,
                enable_practice=False,
                enable_skill_manager=False,
                enable_knowledge=False,
                enable_how=False,
                enable_auto=False,
                enable_provider=False,
                enable_sense=False,
                enable_event_persistence=False,
                enable_tracing=False,
            )
        )
        with DaemonLedger(
            root / "state/ledger.sqlite3", key_path=root / "state/ledger.key"
        ) as ledger:
            service = DaemonControlPlane(runtime=runtime, ledger=ledger, state_dir=root / "state")
            daemon = RosclawDaemon(service=service, socket_path=root / "run/daemon.sock")
            daemon.start()
            try:
                client = DaemonClient(socket_path=root / "run/daemon.sock", timeout_sec=3)
                client.arm_runtime("supplied perception fixture; no robot initialization")
                store = AppStore(home=root / "home")
                installed = store.install(Path(__file__).parent / "apps" / "perception-quality")
                manifest = AppManifest.from_path(installed.path)
                run = (
                    AppRunner(client)
                    .run(
                        manifest,
                        body_id="supplied-fixture",
                        body_snapshot_hash="",
                        execution_mode=ExecutionMode.FIXTURE,
                        inputs={"scan": SCAN_PAYLOAD, "cloud": CLOUD_PAYLOAD},
                    )
                    .to_dict()
                )
                print(json.dumps(run, indent=2, sort_keys=True))
                assert run["status"] == "success"
                assert run["trust_level"] == "SYNTHETIC"
            finally:
                daemon.stop()


if __name__ == "__main__":
    main()
