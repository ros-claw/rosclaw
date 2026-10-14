"""SDK-owned independent observation process, isolated from the ROS callback GIL.

No Runtime, ROS client, lease or action is opened here. The worker reads original
Native/ECM facts and the existing daemon's read-only socket. Its expiring output
is supplementary observation only. Exit, stale output or any failure latches.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from simulation_current_facts import CurrentSimulationFacts, _read
from simulation_navigation_supervision import review_active_sim_daemon
from simulation_source_capture import BoundedSourceJsonlCapture

from rosclaw.daemon.client import DaemonClient


class SdkCurrentFactsProcess:
    def __init__(self, *, root, enrollment_sha256, admission, binding, prepared_process=None):
        self.root = Path(root)
        self.path = self.root / "execution-control/current-monitor.json"
        self.config = {
            "root": str(self.root),
            "enrollment_sha256": enrollment_sha256,
            "admission": admission,
            "body_id": binding.body_id,
        }
        self.process = prepared_process
        self.failed_reason = None
        self._started = False
        self.ready = False
        self._startup_deadline_ns = None

    @staticmethod
    def prepare_process():
        # Import the pure SDK worker during passive fixture bootstrap. It waits
        # on stdin and cannot collect facts or acknowledge admission yet.
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )

    def start(self):
        if self._started or self.failed_reason:
            raise ValueError("SDK current process cannot be restarted")
        self._started = True
        if self.process is None:
            self.process = self.prepare_process()
        self.process.stdin.write(json.dumps(self.config, allow_nan=False).encode() + b"\n")
        self.process.stdin.close()
        self._startup_deadline_ns = time.monotonic_ns() + 500_000_000

    def poll_ready(self):
        if self.failed_reason or self.process is None or self.process.poll() is not None:
            self._fail("SDK independent observation startup unavailable")
        if not self.path.exists():
            if time.monotonic_ns() >= self._startup_deadline_ns:
                self._fail("SDK independent observation startup expired")
            return False
        self.get()
        self.ready = True
        return True

    def _fail(self, reason):
        self.failed_reason = self.failed_reason or str(reason)[:2500]
        raise ValueError("SDK current process permanently latched: " + self.failed_reason)

    def get(self):
        if self.failed_reason:
            self._fail(self.failed_reason)
        try:
            if self.process is None or self.process.poll() is not None:
                raise ValueError("independent observation process exited")
            result = json.loads(_read(self.path, 64_000))
            if (
                set(result) != {"schema_version", "pid", "terminal", "reason", "facts"}
                or result["schema_version"] != "rosclaw.sdk_current_monitor.v1"
                or type(result["pid"]) is not int
                or result["pid"] != self.process.pid
                or result["terminal"] is not False
                or result["reason"] is not None
            ):
                raise ValueError(
                    "independent observation terminal or identity changed: "
                    + str(result.get("reason"))
                )
            facts = result["facts"]
            now = time.monotonic_ns()
            if (
                type(facts["health_review_ns"]) is not int
                or type(facts["valid_until_monotonic_ns"]) is not int
                or not facts["health_review_ns"] <= now < facts["valid_until_monotonic_ns"]
                or now - facts["health_review_ns"] > 500_000_000
            ):
                raise ValueError("independent observation window expired or clock invalid")
            return facts
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._fail(exc)

    def close(self):
        self.failed_reason = self.failed_reason or "SDK_CURRENT_PROCESS_CLOSED"
        self.close_owned_process(self.process)

    @staticmethod
    def close_owned_process(process):
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)


def run_worker(config):
    root = Path(config["root"])
    control = root / "execution-control"
    directory_fd = os.open(control, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    identity = os.fstat(directory_fd)
    facts = None
    audit = None

    def publish(result):
        current = control.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            raise ValueError("original SDK control directory replaced")
        raw = (json.dumps(result, allow_nan=False) + "\n").encode()
        if len(raw) > 64_000:
            raise ValueError("bounded SDK current result exceeded")
        fd = os.open(
            "current-monitor.tmp",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
            os.replace(
                "current-monitor.tmp",
                "current-monitor.json",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        finally:
            with suppress(FileNotFoundError):
                os.unlink("current-monitor.tmp", dir_fd=directory_fd)

    try:
        if set(config) != {"root", "enrollment_sha256", "admission", "body_id"}:
            raise ValueError("closed SDK current process configuration required")
        a = config["admission"]
        facts = CurrentSimulationFacts(
            root, assets_directory=Path("/assets"), enrollment_sha256=config["enrollment_sha256"]
        )
        client = DaemonClient(
            socket_path=control / "daemon/run/rosclawd.sock",
            expected_daemon_uid=a["daemon_uid"],
            timeout_sec=0.1,
        )
        audit = BoundedSourceJsonlCapture(
            root / "sdk-current-monitor-original.jsonl",
            max_segments=1,
            max_bytes=30_000_000,
            max_rows=5000,
            max_line_bytes=64_000,
        )
        while True:
            if time.monotonic_ns() >= a["expires_ns"]:
                raise ValueError("explicit SDK admission expired")
            current = facts.collect(freshness_horizon_ms=100)
            status = client.get_runtime_status()
            if (
                status["daemon_peer"]["pid"] != a["daemon_pid"]
                or status["daemon_peer"]["uid"] != a["daemon_uid"]
                or status["client_peer"]["pid"] != os.getpid()
                or status["process_separated"] is not True
            ):
                raise ValueError("actual independent daemon socket peer changed")
            review_active_sim_daemon(
                status,
                instance_id=a["daemon_instance_id"],
                pid=a["daemon_pid"],
                uid=a["daemon_uid"],
                body_id=config["body_id"],
                expected_running=None,
            )
            current["daemon_status"] = status
            if time.monotonic_ns() >= current["valid_until_monotonic_ns"]:
                raise ValueError("observation expired during actual daemon check")
            result = {
                "schema_version": "rosclaw.sdk_current_monitor.v1",
                "pid": os.getpid(),
                "terminal": False,
                "reason": None,
                "facts": current,
            }
            audit.write(json.dumps(result, allow_nan=False) + "\n")
            publish(result)
            time.sleep(0.01)
    except Exception as exc:
        result = {
            "schema_version": "rosclaw.sdk_current_monitor.v1",
            "pid": os.getpid(),
            "terminal": True,
            "reason": str(exc)[:2500],
            "facts": None,
        }
        if audit is not None:
            audit.write(json.dumps(result, allow_nan=False) + "\n")
        publish(result)
        raise
    finally:
        if facts is not None:
            facts.close()
        if audit is not None:
            audit.close()
        os.close(directory_fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    raw = sys.stdin.buffer.readline(16_001)
    if not raw.endswith(b"\n") or len(raw) > 16_000:
        raise ValueError("bounded closed SDK worker input required")
    run_worker(json.loads(raw))
