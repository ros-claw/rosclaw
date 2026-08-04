"""`rosclaw evidence verify <pack_dir>` — 证据包离线 verifier（二次复核 R7/T6）。

在另一台机器上独立验证 evidence pack：
1. artifact_hashes.json 全部重算匹配；
2. run_manifest.sig 用包外锚验签（与 release 同一锚查找链）；
3. run_manifest 必备字段 + secret_scan_clean；
4. 缺任一关键 artifact 即失败。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "artifact_hashes.json",
    "secret_scan.json",
    "environment.json",
    "commands.txt",
    "events.jsonl",
)


def _anchor_candidates() -> list[str]:
    return [
        os.environ.get("ROSCLAW_RELEASE_KEY", ""),
        "/etc/rosclaw/release-pub.pem",
        str(Path.home() / ".rosclaw" / "release-keys" / "release-pub.pem"),
        str(Path.home() / ".rosclaw" / "signing" / "dev-signing-public.pem"),
    ]


def verify_pack(pack: Path) -> dict:
    errors: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (pack / name).exists():
            errors.append(f"missing artifact: {name}")
    if errors:
        _fail(errors)
    manifest = json.loads((pack / "run_manifest.json").read_text())
    if manifest.get("invalid"):
        _fail([f"pack marked INVALID: {manifest.get('reason', '')}"])
    scan = json.loads((pack / "secret_scan.json").read_text())
    if not scan.get("clean"):
        errors.append(f"secret scan findings: {scan.get('findings')}")
    hashes = json.loads((pack / "artifact_hashes.json").read_text())
    for rel, expected in hashes.items():
        path = pack / rel
        if not path.exists():
            errors.append(f"missing hashed artifact: {rel}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"hash mismatch: {rel}")
    for field in ("run_id", "evidence_level", "rosclaw_commit", "created_at"):
        if not manifest.get(field):
            errors.append(f"run_manifest missing field: {field}")
    # 签名：有 sig 必须验；无 sig 如实标记（开发包可过，发布门禁卡这个）。
    sig = pack / "run_manifest.sig"
    signed = False
    if sig.exists():
        anchor = None
        for candidate in _anchor_candidates():
            if candidate and Path(candidate).exists():
                anchor = Path(candidate)
                break
        if anchor is None:
            errors.append("run_manifest.sig present but no trust anchor available")
        else:
            result = subprocess.run(
                [
                    "openssl", "dgst", "-sha256", "-verify", str(anchor),
                    "-signature", str(sig), str(pack / "run_manifest.json"),
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                errors.append("run_manifest signature invalid")
            else:
                signed = True
    if errors:
        _fail(errors)
    return {
        "verified": True,
        "run_id": manifest["run_id"],
        "evidence_level": manifest["evidence_level"],
        "signed": signed,
        "secret_scan_clean": True,
        "artifacts": len(hashes),
    }


def _fail(errors: list[str]) -> None:
    print("EVIDENCE PACK VERIFICATION FAILED:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(2)


def dispatch_evidence_argv(argv: list[str]) -> int | None:
    if argv[:2] != ["evidence", "verify"]:
        return None
    if len(argv) < 3:
        print("用法: rosclaw evidence verify <pack_dir>", file=sys.stderr)
        return 2
    pack = Path(argv[2])
    if not pack.is_dir():
        print(f"pack 目录不存在: {pack}", file=sys.stderr)
        return 2
    report = verify_pack(pack)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
