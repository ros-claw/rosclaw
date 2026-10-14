"""SDK-only, bounded append-only ECM reader; no ROS or execution authority.

An operator independently pins an accepted stopped G1/L1 prefix and observer
process. Historical prefix validation is not current readiness. Every poll
checks the newest sample against the actual monotonic clock and trips forever
on loss. Closed segment SHA256 values chain into each next segment's header.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from simulation_source_continuity import SimulationSourceContinuityGuard

from rosclaw.connectors.ros.context.navigation_workspace_review import _json


@dataclass(frozen=True)
class SourceStreamPins:
    observer_epoch: str
    observer_pid: int
    segment_max_samples: int
    segment_max_bytes: int
    max_segments: int

    def __post_init__(self):
        if (
            not re.fullmatch(r"[0-9a-f]{32}", self.observer_epoch)
            or type(self.observer_pid) is not int
            or self.observer_pid <= 0
            or type(self.segment_max_samples) is not int
            or not 1 <= self.segment_max_samples <= 600
            or type(self.segment_max_bytes) is not int
            or not 500_000 <= self.segment_max_bytes <= 30_000_000
            or type(self.max_segments) is not int
            or not 1 <= self.max_segments <= 64
            or self.segment_max_bytes * self.max_segments > 120_000_000
        ):
            raise ValueError("independent bounded source stream pins required")

    def header(self, index, first_sample, previous_digest):
        return {
            "schema_version": "rosclaw.ecm_segment.v1",
            "observer_epoch": self.observer_epoch,
            "observer_pid": self.observer_pid,
            "segment_index": index,
            "segment_first_sample": first_sample,
            "previous_segment_sha256": previous_digest,
            "segment_max_samples": self.segment_max_samples,
            "segment_max_bytes": self.segment_max_bytes,
            "max_segments": self.max_segments,
            "sample_interval_ms": 100,
        }


class SimulationSourceStreamReader:
    """Terminal observer with explicit close; accepted prefix must be segment 0.

    This first implementation intentionally refuses admission after rollover:
    operator recapture must start a new stream. It never silently skips a past
    segment or treats a cursor from another process as current trust.
    """

    def __init__(
        self,
        directory: Path,
        basename: str,
        *,
        pins: SourceStreamPins,
        accepted_prefix: bytes,
        expected_prefix_sha256: str,
        guard_inputs: dict,
        now_monotonic_ns: int | None = None,
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.jsonl", basename):
            raise ValueError("single explicit source basename required")
        if (
            type(accepted_prefix) is not bytes
            or not accepted_prefix.endswith(b"\n")
            or not 0 < len(accepted_prefix) <= pins.segment_max_bytes
            or hashlib.sha256(accepted_prefix).hexdigest() != expected_prefix_sha256
        ):
            raise ValueError("separate original complete prefix SHA256 required")
        self.pins = pins
        self.basename = basename
        self.directory = Path(directory)
        self.directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.directory_identity = self._identity(os.fstat(self.directory_fd))
        self.files = []
        self.closed = False
        self.failed_reason = None
        self.index = self.offset = self.first_sample = self.count = 0
        self.previous_digest = ""
        self.last_sample = -1
        self.last_raw = b""
        self.guard = None
        self.digest = hashlib.sha256()
        self.segment_chain = []
        try:
            self._open_segment()
            if os.pread(self.files[0][0], len(accepted_prefix), 0) != accepted_prefix:
                raise ValueError("accepted source prefix does not match original file")
            lines = accepted_prefix.splitlines(keepends=True)
            for raw in lines:
                self._consume(raw, before_admission=True)
            reference = _json(self.last_raw)
            self.guard = SimulationSourceContinuityGuard.from_pinned_sample(
                self.last_raw,
                expected_sha256=hashlib.sha256(self.last_raw).hexdigest(),
                now_monotonic_ns=reference["observer_monotonic_ns"],
                **guard_inputs,
            )
            self.offset = len(accepted_prefix)
            self.digest.update(accepted_prefix)
            if not self.poll(now_monotonic_ns=now_monotonic_ns)["continuous"]:
                raise ValueError(self.failed_reason)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _identity(s):
        return s.st_dev, s.st_ino

    def _name(self, index):
        return self.basename if index == 0 else f"{self.basename}.segment-{index:06}.jsonl"

    def _open_segment(self):
        fd = os.open(self._name(self.index), os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.directory_fd)
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode):
            os.close(fd)
            raise ValueError("regular original source segment required")
        self.files.append((fd, self._identity(s), None))

    def _consume(self, raw, *, before_admission=False):
        if not raw.endswith(b"\n") or not 0 < len(raw) <= 500_000:
            raise ValueError("complete bounded source line required")
        sample = _json(raw)
        if type(sample["sample"]) is not int or sample["sample"] != self.last_sample + 1:
            raise ValueError("source sample cursor gap or regression")
        if json.dumps(sample["source_stream"], sort_keys=True, allow_nan=False) != json.dumps(
            self.pins.header(self.index, self.first_sample, self.previous_digest), sort_keys=True
        ):
            raise ValueError("source epoch/process/budget/segment/hash-chain drift")
        if self.count >= self.pins.segment_max_samples:
            raise ValueError("source segment sample overflow")
        if not before_admission:
            result = self.guard.observe(raw, now_monotonic_ns=sample["observer_monotonic_ns"])
            if not result["continuous"]:
                raise ValueError(result["reason"])
        self.last_raw = raw
        self.last_sample = sample["sample"]
        self.count += 1

    def poll(self, *, now_monotonic_ns: int | None = None):
        reviewed_now = now_monotonic_ns
        if self.failed_reason is None:
            try:
                if self.closed:
                    raise ValueError("source reader closed")
                if (
                    self._identity(os.stat(self.directory, follow_symlinks=False))
                    != self.directory_identity
                ):
                    raise ValueError("source directory replaced")
                for i, (_fd, identity, sealed_stat) in enumerate(self.files):
                    current = os.stat(
                        self._name(i), dir_fd=self.directory_fd, follow_symlinks=False
                    )
                    if self._identity(current) != identity or not stat.S_ISREG(current.st_mode):
                        raise ValueError("source segment replaced")
                    if (
                        sealed_stat is not None
                        and (current.st_size, current.st_mtime_ns, current.st_ctime_ns)
                        != sealed_stat
                    ):
                        raise ValueError("sealed source segment mutated")
                while True:
                    fd = self.files[self.index][0]
                    s = os.fstat(fd)
                    if not self.offset <= s.st_size <= self.pins.segment_max_bytes:
                        raise ValueError("source segment truncated or overflowed")
                    prefix = os.pread(fd, self.offset, 0)
                    if hashlib.sha256(prefix).digest() != self.digest.digest():
                        raise ValueError("original consumed source prefix rewritten")
                    tail = os.pread(fd, s.st_size - self.offset, self.offset)
                    complete = tail.rfind(b"\n") + 1
                    if len(tail) - complete > 500_000:
                        raise ValueError("source partial line overflow")
                    for raw in tail[:complete].splitlines(keepends=True):
                        self._consume(raw)
                    self.digest.update(tail[:complete])
                    self.offset += complete
                    next_name = self._name(self.index + 1)
                    names = os.listdir(self.directory_fd)
                    if len(names) > 4096:
                        raise ValueError("bounded source directory inventory required")
                    for name in names:
                        if name.startswith(self.basename + ".segment-"):
                            suffix = name[len(self.basename + ".segment-") :]
                            if not re.fullmatch(r"[0-9]{6}\.jsonl", suffix):
                                raise ValueError("invalid source segment name")
                            number = int(suffix[:6])
                            if (
                                number == 0
                                or number >= self.pins.max_segments
                                or any(
                                    self._name(n) not in names
                                    for n in range(self.index + 1, number + 1)
                                )
                            ):
                                raise ValueError("source segment gap or capacity overflow")
                    if next_name not in names:
                        break
                    if complete != len(tail) or not self.count:
                        raise ValueError("rollover left an incomplete source segment")
                    digest = self.digest.hexdigest()
                    self.segment_chain.append(
                        {
                            "index": self.index,
                            "sha256": digest,
                            "bytes": self.offset,
                            "samples": self.count,
                            "previous_sha256": self.previous_digest,
                        }
                    )
                    self.files[self.index] = (
                        fd,
                        self.files[self.index][1],
                        (s.st_size, s.st_mtime_ns, s.st_ctime_ns),
                    )
                    self.previous_digest = digest
                    self.index += 1
                    self.first_sample = self.last_sample + 1
                    self.offset = self.count = 0
                    self.digest = hashlib.sha256()
                    self._open_segment()
                # Read first, then sample the live clock: a writer may append a
                # new record during this poll. Explicit timestamps are replay
                # and contract inputs only; never use a pre-read live time.
                reviewed_now = time.monotonic_ns() if reviewed_now is None else reviewed_now
                result = self.guard.observe(self.last_raw, now_monotonic_ns=reviewed_now)
                if not result["continuous"]:
                    raise ValueError(result["reason"])
                if self.index + 1 == self.pins.max_segments and (
                    self.count >= self.pins.segment_max_samples
                    or self.offset + 500_000 > self.pins.segment_max_bytes
                ):
                    raise ValueError("source evidence capacity exhausted; recapture required")
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                RecursionError,
                OverflowError,
            ) as exc:
                self.failed_reason = str(exc)
        return {
            "schema_version": "rosclaw.sim_source_stream_review.v1",
            "continuous": self.failed_reason is None,
            "terminal": self.failed_reason is not None,
            "reason": self.failed_reason,
            "reviewed_at_monotonic_ns": time.monotonic_ns()
            if reviewed_now is None
            else reviewed_now,
            "last_sample": self.last_sample,
            "segment_index": self.index,
            "segment_chain": list(self.segment_chain),
            "execution_authority": "NONE",
            "L1": "NOT_EVALUATED",
            "L2": "NOT_RUN",
        }

    def capacity_covers(self, *, duration_ms: int, max_sample_bytes: int):
        """Conservative evidence budget only, never admission/permission."""
        if (
            type(duration_ms) is not int
            or duration_ms <= 0
            or type(max_sample_bytes) is not int
            or not 1 <= max_sample_bytes <= 500_000
        ):
            raise ValueError("bounded independent duration and record-size bounds required")
        # Reserve one extra sample for dispatch/rollover, and at most one lost
        # record per future segment due to a record crossing its byte boundary.
        needed = (duration_ms + 99) // 100 + 1
        future = self.pins.max_segments - self.index - 1
        samples = (
            self.pins.segment_max_samples - self.count + future * self.pins.segment_max_samples
        )
        records = max(0, (self.pins.segment_max_bytes - self.offset) // max_sample_bytes)
        records += future * max(0, self.pins.segment_max_bytes // max_sample_bytes - 1)
        return not self.closed and self.failed_reason is None and needed <= min(samples, records)

    def close(self):
        if not self.closed:
            for fd, _, _ in self.files:
                os.close(fd)
            os.close(self.directory_fd)
            self.closed = True
