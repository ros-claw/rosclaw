"""Bounded exclusive raw contact segments; no ROS, actuator or authority."""

import hashlib
import json
import os
from pathlib import Path


class BoundedSourceJsonlCapture:
    def __init__(
        self,
        path: Path,
        *,
        max_segments: int = 1,
        max_bytes: int = 30_000_000,
        max_rows: int = 20_000,
        max_line_bytes: int = 500_000,
    ):
        if (
            type(max_segments) is not int
            or not 1 <= max_segments <= 4
            or type(max_bytes) is not int
            or not 1 <= max_bytes <= 30_000_000
            or type(max_rows) is not int
            or not 1 <= max_rows <= 20_000
            or type(max_line_bytes) is not int
            or not 1 <= max_line_bytes <= 500_000
        ):
            raise ValueError("explicit bounded raw contact evidence capacity required")
        self.path = Path(path)
        self.max_segments, self.max_bytes = max_segments, max_bytes
        self.max_rows, self.max_line_bytes = max_rows, max_line_bytes
        self.index = self.rows = self.bytes = 0
        self.previous = ""
        self.digest = hashlib.sha256()
        self.closed = False
        self.file = self._open(self.path)
        self.chain = None
        try:
            if max_segments > 1:
                self.chain = self._open(self.path.with_name(self.path.stem + ".chain.jsonl"))
        except BaseException:
            self.file.close()
            raise

    @staticmethod
    def _open(path):
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
        return os.fdopen(fd, "wb", buffering=0)

    def _seal(self):
        self.file.flush()
        os.fdatasync(self.file.fileno())
        self.file.close()
        digest = self.digest.hexdigest()
        if self.chain is not None:
            record = {
                "schema_version": "rosclaw.raw_contact_segment.v1",
                "index": self.index,
                "file": Path(self.file.name).name
                if isinstance(self.file.name, str)
                else self._name().name,
                "sha256": digest,
                "previous_sha256": self.previous,
                "bytes": self.bytes,
                "rows": self.rows,
            }
            self._write_all(self.chain, (json.dumps(record, sort_keys=True) + "\n").encode())
            os.fdatasync(self.chain.fileno())
        self.previous = digest

    def _name(self):
        return (
            self.path
            if self.index == 0
            else self.path.with_name(self.path.stem + f".segment-{self.index:06}.jsonl")
        )

    @staticmethod
    def _write_all(file, raw):
        view = memoryview(raw)
        while view:
            n = file.write(view)
            if not n:
                raise OSError("partial raw source write failed")
            view = view[n:]

    def write(self, line: str):
        raw = line.encode("utf-8")
        if self.closed:
            raise RuntimeError("raw contact capture closed")
        if (
            not raw.endswith(b"\n")
            or b"\n" in raw[:-1]
            or not 0 < len(raw) <= min(self.max_line_bytes, self.max_bytes)
        ):
            self.close()
            raise RuntimeError("bounded complete raw contact line required")
        if self.rows >= self.max_rows or self.bytes + len(raw) > self.max_bytes:
            self._seal()
            self.index += 1
            if self.index >= self.max_segments:
                self.closed = True
                if self.chain is not None:
                    self.chain.close()
                raise RuntimeError("bounded original contact capture exhausted")
            self.rows = self.bytes = 0
            self.digest = hashlib.sha256()
            try:
                self.file = self._open(self._name())
            except BaseException:
                self.closed = True
                if self.chain is not None:
                    self.chain.close()
                raise
        try:
            self._write_all(self.file, raw)
            self.digest.update(raw)
            self.rows += 1
            self.bytes += len(raw)
        except BaseException:
            self.closed = True
            self.file.close()
            if self.chain is not None:
                self.chain.close()
            raise

    def close(self):
        if not self.closed:
            try:
                self._seal()
            finally:
                self.closed = True
                if self.chain is not None:
                    self.chain.close()
