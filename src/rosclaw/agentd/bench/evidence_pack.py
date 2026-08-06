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

# R7/T6：secret corpus——命中即 fail（不是 warning）。
SECRET_PATTERNS = (
    "sk-",
    "sk-ant-",
    "sk-kimi-",
    "ghp_",
    "gho_",
    "github_pat_",
    "Bearer ",
    "eyJ",  # JWT header 前缀
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN EC PRIVATE KEY",
    "BEGIN PRIVATE KEY",
    "api_key\":",
    "private_key\":",
    "permit_secret",
    "refresh_token",
)


class EvidencePackError(RuntimeError):
    pass


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
        # R7：自动生成的观察记录必须明确标注——不是独立人类观察者签字。
        self._record(
            "automated_observation.md",
            "> **automated_observation**（自动生成，非独立观察者签字）\n\n" + note,
        )

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
            if not path.is_file() or path.name in {
                "secret_scan.json",
                "run_manifest.json",
                "run_manifest.sig",
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_PATTERNS:
                if pattern in text:
                    findings.append({"file": path.name, "pattern": pattern})
        self._record(
            "secret_scan.json",
            json.dumps({"clean": not findings, "findings": findings}, indent=2),
        )
        if findings:
            # R7/T6：发现 secret 即 fail——证据包标记 INVALID 并拒绝 finalize。
            self._record(
                "run_manifest.json",
                json.dumps(
                    {
                        "run_id": self.run_id,
                        "invalid": True,
                        "reason": "secret_scan_findings",
                        "findings": findings,
                    },
                    indent=2,
                ),
            )
            raise EvidencePackError(
                f"evidence pack contains secret-like material: {findings} — "
                "pack marked INVALID; redact sources and re-run"
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
            "process": {"uid": os.geteuid(), "pid": os.getpid()},
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._record("run_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        # R7：pack 签名（构建机 dev release key；无 key 时如实标 unsigned）。
        signed = self._try_sign_manifest()
        manifest["signed"] = signed
        self._hashes["run_manifest.json"] = hashlib.sha256(
            (self.dir / "run_manifest.json").read_bytes()
        ).hexdigest()
        # artifact_hashes.json 不包含自己的条目（否则自引用永真失配）。
        final_hashes = {k: v for k, v in self._hashes.items() if k != "artifact_hashes.json"}
        self._record(
            "artifact_hashes.json",
            json.dumps(final_hashes, indent=2, sort_keys=True),
        )
        return manifest

    def _try_sign_manifest(self) -> bool:
        """用本机 release 签名 key 签 run_manifest.json（openssl ECDSA）。

        无签名 key 时返回 False（pack 仍可用于开发调试，但不能作为
        可审计发布证据——verifier 会要求签名）。
        """
        import shutil
        import subprocess

        key = Path.home() / ".rosclaw" / "signing" / "dev-signing-private.pem"
        if not key.exists() or shutil.which("openssl") is None:
            return False
        result = subprocess.run(
            [
                "openssl",
                "dgst",
                "-sha256",
                "-sign",
                str(key),
                "-out",
                str(self.dir / "run_manifest.sig"),
                str(self.dir / "run_manifest.json"),
            ],
            capture_output=True,
        )
        return result.returncode == 0


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
