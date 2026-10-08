"""Bounded append cursor for fixture orchestration, never final-audit acceptance.

Read from the beginning of one exclusive live audit. An unfinished trailing
row waits for completion; a bad completed row, gap, mutation or replacement
latches a fault. The physical observer remains the source of each packet.
"""

import json
import os
from pathlib import Path

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


class AuditCursor:
    def __init__(self, path, *, run_id, read_limit=2_000_000):
        if (
            type(run_id) is not str
            or not run_id
            or type(read_limit) is not int
            or not 1024 <= read_limit <= 2_000_000
        ):
            raise ValueError("bounded cursor and frozen run required")
        self.path = Path(path)
        self.run_id = run_id
        self.limit = read_limit
        self.offset = 0
        self.pending = b""
        self.sequence = 0
        self.previous_hash = None
        self.identity = None
        self.fault = None

    def poll(self):
        if self.fault is not None:
            raise ValueError("live audit cursor fault is latched: " + self.fault)
        try:
            with self.path.open("rb") as stream:
                info = os.fstat(stream.fileno())
                identity = (info.st_dev, info.st_ino)
                if self.identity is None:
                    self.identity = identity
                if (
                    identity != self.identity
                    or info.st_size < self.offset
                    or info.st_size > 1_000_000_000
                ):
                    raise ValueError("live audit replaced/truncated or exceeds bound")
                if info.st_size - self.offset > self.limit:
                    raise ValueError("live audit unread backlog exceeds cursor bound")
                stream.seek(self.offset)
                block = stream.read(self.limit - len(self.pending))
                current = self.path.stat()
                if (current.st_dev, current.st_ino) != identity:
                    raise ValueError("live audit replaced during read")
            self.offset += len(block)
            data = self.pending + block
            if len(data) > self.limit:
                raise ValueError("live audit incomplete row exceeds bound")
            complete, newline, self.pending = data.rpartition(b"\n")
            if not newline:
                self.pending = data
                return []
            rows = []
            for line in complete.split(b"\n"):
                row = json.loads(line)
                if type(row) is not dict:
                    raise ValueError("completed live audit row must be an object")
                saved = row.pop("artifact_sha256", None)
                if (
                    row.get("schema_version") != "rosclaw.coverage_audit_event.v1"
                    or row.get("run_id") != self.run_id
                    or type(row.get("sequence")) is not int
                    or row["sequence"] != self.sequence + 1
                    or row.get("previous_hash") != self.previous_hash
                    or digest(row) != saved
                ):
                    raise ValueError("completed live audit identity/hash/sequence mismatch")
                self.sequence = row["sequence"]
                self.previous_hash = saved
                row["artifact_sha256"] = saved
                rows.append(row)
            return rows
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            self.fault = str(exc)
            raise ValueError("invalid live fixture audit: " + self.fault) from exc

    def status(self):
        return {
            "evidence_role": "LIVE_PREFIX_NOT_FINAL_AUDIT",
            "run_id": self.run_id,
            "sequence": self.sequence,
            "pending_bytes": len(self.pending),
            "fault": self.fault,
            "physical_acceptance": "NOT_RUN",
        }
