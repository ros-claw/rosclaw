"""Artifact result store (PR-05, 大纲 §7.5).

Oversized tool outputs (ROS bags, images-as-metadata, long telemetry) are
spilled to content-addressed files under the agentd home; the model only
ever sees the ref + digest + head excerpt. Nothing is silently truncated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class ArtifactResultStore:
    def __init__(self, root: Path) -> None:
        self._root = root / "artifacts" / "observations"
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, content: str, *, prefix: str = "observation") -> str:
        digest = hashlib.sha256(content.encode()).hexdigest()
        path = self._root / f"{prefix}-{digest}.txt"
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        return f"artifact://{prefix}/sha256:{digest}"

    def resolve(self, ref: str) -> str | None:
        """Read back a stored artifact by ref; None if unknown (fail closed)."""
        if not ref.startswith("artifact://") or "/sha256:" not in ref:
            return None
        prefix = ref[len("artifact://") :].split("/sha256:", 1)[0]
        digest = ref.rsplit("/sha256:", 1)[1]
        path = self._root / f"{prefix}-{digest}.txt"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
