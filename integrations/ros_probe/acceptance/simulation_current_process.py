"""SDK-owned independent observation process, isolated from the ROS callback GIL.

No Runtime, ROS client, lease or action is opened here. The worker reads original
Native/ECM facts and the existing daemon's read-only socket. Its expiring output
is supplementary observation only. Exit, stale output or any failure latches.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

from simulation_current_facts import CurrentSimulationFacts, _read
from simulation_daemon_observation import BoundedDaemonObservation
from simulation_source_capture import BoundedSourceJsonlCapture

from rosclaw.daemon.client import DaemonClient


class SdkPassiveFactsPreparation:
    """Prepare readonly collector inputs without blocking the sink heartbeat."""

    def __init__(self, root, *, enrollment_sha256, body_hash, run_id):
        self.root = root
        self.digest, self.body_hash, self.run_id = enrollment_sha256, body_hash, run_id
        self.result = self.error = self.worker = None

    def start(self):
        if self.worker is not None:
            raise ValueError("passive preparation cannot restart")

        def build():
            try:
                facts = CurrentSimulationFacts(
                    self.root, assets_directory=Path("/assets"), enrollment_sha256=self.digest
                )
                pins = facts.enrollment["request"]
                if (
                    pins["body_snapshot_hash"] != self.body_hash
                    or pins["sim_run_id"] != self.run_id
                ):
                    facts.close()
                    raise ValueError("passive facts differ from the current unadmitted Body/run")
                self.result = facts
            except Exception as exc:
                self.error = str(exc)[:1000]

        self.worker = threading.Thread(
            target=build, daemon=True, name="passive-sdk-facts-preparation"
        )
        self.worker.start()

    def take_if_ready(self):
        if self.worker is None or self.worker.is_alive():
            return None
        if self.error is not None:
            raise ValueError("passive preparation failed: " + self.error)
        result, self.result = self.result, None
        if result is None:
            raise ValueError("passive preparation result unavailable or already consumed")
        return result

    def close(self):
        if self.worker is not None:
            self.worker.join(timeout=2)
        if self.result is not None:
            self.result.close()
            self.result = None


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
            for attempt in range(2):
                result = json.loads(_read(self.path, 64_000, allow_atomic_unlink=True))
                if (
                    set(result)
                    != {
                        "schema_version",
                        "pid",
                        "terminal",
                        "reason",
                        "facts",
                        "observed_monotonic_ns",
                    }
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
                    type(result["observed_monotonic_ns"]) is not int
                    or not facts["completed_ns"] <= result["observed_monotonic_ns"] <= now
                    or type(facts["health_review_ns"]) is not int
                    or type(facts["valid_until_monotonic_ns"]) is not int
                    or not facts["health_review_ns"] <= now < facts["valid_until_monotonic_ns"]
                    or now - facts["health_review_ns"] > 500_000_000
                ):
                    if attempt == 0 and now >= facts["valid_until_monotonic_ns"]:
                        # The worker atomically replaces this file. A reader can
                        # hold the old inode across scheduling while a newer
                        # valid observation is already published. Reopen ONCE;
                        # never return the expired snapshot or wait out loss.
                        continue
                    raise ValueError("independent observation window expired or clock invalid")
                return facts
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._fail(exc)

    def close(self):
        self.failed_reason = self.failed_reason or "SDK_CURRENT_PROCESS_CLOSED"
        self.close_owned_process(self.process)

    def get_at_least_source(self, sample):
        if type(sample) is not int or sample < 0:
            self._fail("authenticated source cursor required")
        deadline = time.monotonic_ns() + 100_000_000
        while True:
            facts = self.get()
            if facts["source"]["last_sample"] >= sample:
                return facts
            if time.monotonic_ns() >= deadline:
                self._fail("independent source cursor did not catch up within 100ms")
            time.sleep(0.002)

    @staticmethod
    def close_owned_process(process):
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def run_worker(config):
    root = Path(config["root"])
    control = root / "execution-control"
    directory_fd = os.open(control, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    identity = os.fstat(directory_fd)
    facts = None
    audit = None
    startup = None
    daemon_monitor = None

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
        startup = BoundedSourceJsonlCapture(
            root / "sdk-current-startup-original.jsonl",
            max_segments=1,
            max_bytes=64_000,
            max_rows=12,
            max_line_bytes=4096,
        )

        def stage(name):
            startup.write(
                json.dumps({"stage": name, "pid": os.getpid(), "monotonic_ns": time.monotonic_ns()})
                + "\n"
            )

        stage("CONFIGURATION_ACCEPTED")
        a = config["admission"]
        facts = CurrentSimulationFacts(
            root, assets_directory=Path("/assets"), enrollment_sha256=config["enrollment_sha256"]
        )
        stage("CURRENT_FACTS_CONSTRUCTED")
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
        daemon_monitor = BoundedDaemonObservation(
            client=client, admission=a, body_id=config["body_id"]
        )
        stage("DAEMON_OBSERVER_STARTING")
        daemon_monitor.start()
        stage("DAEMON_OBSERVER_STARTED")
        first = True
        while True:
            if first:
                stage("FIRST_FACTS_BEGIN")
            current = facts.collect(freshness_horizon_ms=100)
            if first:
                stage("FIRST_FACTS_COLLECTED")
            daemon = daemon_monitor.snapshot()
            current["daemon_status"] = daemon["status"]
            current["daemon_review_started_ns"] = daemon["started_ns"]
            current["daemon_review_completed_ns"] = daemon["completed_ns"]
            current["valid_until_monotonic_ns"] = min(
                current["valid_until_monotonic_ns"], daemon["valid_until_ns"]
            )
            current["worker_publish_prepared_ns"] = time.monotonic_ns()
            if time.monotonic_ns() >= current["valid_until_monotonic_ns"]:
                raise ValueError("observation expired during actual daemon check")
            result = {
                "schema_version": "rosclaw.sdk_current_monitor.v1",
                "pid": os.getpid(),
                "observed_monotonic_ns": time.monotonic_ns(),
                "terminal": False,
                "reason": None,
                "facts": current,
            }
            audit.write(json.dumps(result, allow_nan=False) + "\n")
            publish(result)
            if first:
                stage("FIRST_RESULT_PUBLISHED")
                first = False
            time.sleep(0.01)
    except Exception as exc:
        if startup is not None:
            stage("FAILED_NO_CURRENT_ADMISSION")
        result = {
            "schema_version": "rosclaw.sdk_current_monitor.v1",
            "pid": os.getpid(),
            "observed_monotonic_ns": time.monotonic_ns(),
            "terminal": True,
            "reason": str(exc)[:2500],
            "facts": None,
        }
        if audit is not None:
            audit.write(json.dumps(result, allow_nan=False) + "\n")
        publish(result)
        raise
    finally:
        if daemon_monitor is not None:
            daemon_monitor.close()
        if facts is not None:
            facts.close()
        if audit is not None:
            audit.close()
        if startup is not None:
            startup.close()
        os.close(directory_fd)


if __name__ == "__main__":

    def shutdown_owned_worker(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown_owned_worker)
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    # Complete bounded immutable mesh parsing during passive preparation. No
    # current enrollment, daemon observation or startup acknowledgment exists
    # yet; fresh resource/source checks still run after the input arrives.
    from simulation_collision_resources import prewarm_prepared_collision_facts

    prewarm_prepared_collision_facts()
    raw = sys.stdin.buffer.readline(16_001)
    if not raw.endswith(b"\n") or len(raw) > 16_000:
        raise ValueError("bounded closed SDK worker input required")
    run_worker(json.loads(raw))
