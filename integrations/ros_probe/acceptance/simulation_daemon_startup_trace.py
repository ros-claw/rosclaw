"""Optional bounded SDK diagnostic, no new request, retry or execution path."""

import json
import os
import resource
import threading
import time
from pathlib import Path

from simulation_source_capture import BoundedSourceJsonlCapture


class DaemonStartupTrace:
    def __init__(self, daemon, directory):
        if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
            raise ValueError("owned isolated SDK required for daemon startup diagnostics")
        self.daemon = daemon
        self.lock = threading.Lock()
        self.closed = False
        self.audit = BoundedSourceJsonlCapture(
            Path(directory) / "sdk-daemon-startup-original.jsonl",
            max_segments=1,
            max_bytes=256_000,
            max_rows=300,
            max_line_bytes=4096,
        )
        self.original = threading.Thread.start

        def start(thread, *args, **kwargs):
            try:
                return self.original(thread, *args, **kwargs)
            except Exception as exc:
                if thread.name == "rosclawd-client":
                    self.write("CLIENT_THREAD_START_FAILED", error=str(exc)[:1000])
                raise

        self.wrapper = start
        threading.Thread.start = start

    def write(self, event, **fields):
        with self.lock:
            if not self.closed:
                self.audit.write(
                    json.dumps(
                        {
                            "event": event,
                            "monotonic_ns": time.monotonic_ns(),
                            "pid": os.getpid(),
                            **fields,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )

    def sample(self):
        d = self.daemon
        with d._lock:
            clients = len(d._client_threads)
        self.write(
            "READONLY_SERVER_CAPACITY",
            client_threads=clients,
            client_slots_available=d._client_slots._value,
            max_clients=d.max_clients,
            nproc_limit=list(resource.getrlimit(resource.RLIMIT_NPROC)),
            nofile_limit=list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        )

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                if threading.Thread.start is self.wrapper:
                    threading.Thread.start = self.original
                self.audit.close()
