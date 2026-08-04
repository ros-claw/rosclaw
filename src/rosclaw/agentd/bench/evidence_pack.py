"""Evidence pack writer（审计 §8）：每个验收 run 一个 acceptance/<run_id>/。

通过标准：没有证据包，就没有通过。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.agentd.bench.evidence_levels import EvidenceLevel
from rosclaw.contracts.common import new_id

SECRET_PATTERNS = ("sk-", "Bearer ", "api_key\":", "private_key", "permit_secret")


class EvidencePackWriter:
    def __init__(self, root: Path, *, run_id: str | None = None) -> None:
        self.run_id = run_id or new_id("run")
        self.dir = root / "acceptance" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._hashes: dict[str, str] = {}

    def _record(self, name: str, content: str) -> None:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._hashes[name] = hashlib.sha256(content.encode()).hexdigest()

    def write_environment(self, *, provider: str = "", model: str = "") -> dict:
        env = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "node": _probe(["node", "--version"]),
            "ros_distro": os.environ.get("ROS_DISTRO", ""),
            "provider": provider,
            "model": model,
            "uid": os.geteuid(),
        }
        self._record("environment.json", json.dumps(env, indent=2, ensure_ascii=False))
        return env

    def write_commands(self, commands: list[str]) -> None:
        self._record("commands.txt", "\n".join(commands) + "\n")

    def write_events(self, events: list[dict]) -> None:
        lines = [json.dumps(e, ensure_ascii=False, default=str) for e in events]
        self._record("events.jsonl", "\n".join(lines) + ("\n" if lines else ""))

    def write_mission_snapshot(self, snapshot: dict) -> None:
        self._record("mission_snapshot.json", json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))

    def write_public_records(self, *, approvals: list[dict], permits: list[dict], receipts: list[dict]) -> None:
        self._record(
            "approvals_public.jsonl",
            "\n".join(json.dumps(a, ensure_ascii=False, default=str) for a in approvals) + "\n",
        )
        self._record(
            "permits_public.jsonl",
            "\n".join(json.dumps(p, ensure_ascii=False, default=str) for p in permits) + "\n",
        )
        self._record(
            "receipts.jsonl",
            "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in receipts) + "\n",
        )

    def write_metrics(self, metrics: dict) -> None:
        self._record("metrics.json", json.dumps(metrics, indent=2, ensure_ascii=False, default=str))

    def write_observer(self, note: str) -> None:
        self._record("operator_observer.md", note)

    def finalize(
        self,
        *,
        level: EvidenceLevel,
        git_commit: str,
        dirty: bool,
        test_ids: list[str],
        operator: str,
    ) -> dict:
        """secret 扫描 + artifact hashes + run_manifest（签名输入）。"""
        findings: list[dict] = []
        for path in self.dir.rglob("*"):
            if not path.is_file() or path.name in {"secret_scan.json", "run_manifest.json"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_PATTERNS:
                if pattern in text:
                    findings.append({"file": path.name, "pattern": pattern})
        self._record(
            "secret_scan.json",
            json.dumps({"clean": not findings, "findings": findings}, indent=2),
        )
        self._record(
            "artifact_hashes.json",
            json.dumps(self._hashes, indent=2, sort_keys=True),
        )
        manifest = {
            "run_id": self.run_id,
            "evidence_level": level.value,
            "rosclaw_commit": git_commit,
            "dirty": dirty,
            "test_ids": test_ids,
            "operator": operator,
            "started_artifacts": len(self._hashes),
            "secret_scan_clean": not findings,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._record("run_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest


def _probe(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=10).strip()
    except Exception:  # noqa: BLE001
        return ""


def current_commit() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True, timeout=10).strip()
        )
        return commit, dirty
    except Exception:  # noqa: BLE001
        return "", True
