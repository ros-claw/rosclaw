"""`rosclaw release verify <bundle>` — 离线发布包验证 CLI（二次复核 R6）。

在另一台机器上独立验证发布包：
1. 信任锚必须来自包外（--trusted-key / $ROSCLAW_RELEASE_KEY /
   /etc/rosclaw/release-pub.pem / ~/.rosclaw/release-keys/release-pub.pem）；
2. openssl 验证 manifest 签名；
3. 全文件 sha256 + 额外文件拒绝 + symlink 拒绝。

退出码：0=VERIFIED；2=拒绝（任何一项不满足）。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ALLOWED_EXTRA = {"manifest.json", "manifest.sig", "bundle-signing-public.pem"}


def _anchor_candidates() -> list[str]:
    return [
        os.environ.get("ROSCLAW_RELEASE_KEY", ""),
        "/etc/rosclaw/release-pub.pem",
        str(Path.home() / ".rosclaw" / "release-keys" / "release-pub.pem"),
    ]


def verify_bundle(
    bundle: Path,
    *,
    trusted_key: Path | None = None,
    trusted_fingerprint: str = "",
) -> dict:
    """验证一个解包目录或 tar.gz。返回验证报告 dict；失败抛 SystemExit(2)。"""
    errors: list[str] = []
    warnings: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        if bundle.is_dir():
            root = bundle
        else:
            root = Path(tmp) / "bundle"
            with tarfile.open(bundle) as tar:
                # 解包即防线：拒绝绝对路径/.. 成员。
                for member in tar.getmembers():
                    name = member.name
                    if name.startswith("/") or ".." in Path(name).parts:
                        errors.append(f"unsafe tar member: {name}")
                if not errors:
                    tar.extractall(root, filter="data")
            subs = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
            if len(subs) == 1:
                root = subs[0]
        if errors:
            _fail(errors, warnings)
        manifest_path = root / "manifest.json"
        sig_path = root / "manifest.sig"
        if not manifest_path.exists():
            errors.append("manifest.json missing")
        if not sig_path.exists():
            errors.append("manifest.sig missing")
        if errors:
            _fail(errors, warnings)

        anchor = trusted_key
        if anchor is None:
            for candidate in _anchor_candidates():
                if candidate and Path(candidate).exists():
                    anchor = Path(candidate)
                    break
        if anchor is None:
            _fail(
                errors
                + [
                    "no out-of-bundle trust anchor (use --trusted-key or pre-seed "
                    "~/.rosclaw/release-keys/release-pub.pem)"
                ],
                warnings,
            )
        if trusted_fingerprint:
            proc = subprocess.run(
                ["openssl", "pkey", "-pubin", "-in", str(anchor), "-outform", "DER"],
                capture_output=True,
            )
            actual = hashlib.sha256(proc.stdout).hexdigest()
            if actual != trusted_fingerprint:
                _fail(errors + [f"anchor fingerprint mismatch: {actual}"], warnings)
        bundled_key = root / "bundle-signing-public.pem"
        if bundled_key.exists() and bundled_key.read_bytes() != anchor.read_bytes():
            warnings.append("bundle key differs from anchor — anchor wins (possible repackaging)")

        verify = subprocess.run(
            [
                "openssl", "dgst", "-sha256", "-verify", str(anchor),
                "-signature", str(sig_path), str(manifest_path),
            ],
            capture_output=True,
            text=True,
        )
        if verify.returncode != 0:
            _fail(errors + ["manifest signature invalid"], warnings)

        manifest = json.loads(manifest_path.read_text())
        for rel, expected in manifest["files"].items():
            path = root / rel
            if not path.exists():
                errors.append(f"missing: {rel}")
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"tampered: {rel}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                # 包内相对 symlink（bundled node shims）允许；逃逸/悬空链接拒绝。
                target = path.resolve()
                if not str(target).startswith(str(root.resolve())) or not target.exists():
                    errors.append(f"unsafe symlink: {path.relative_to(root)}")
            elif path.is_file():
                rel = str(path.relative_to(root))
                if rel not in manifest["files"] and rel not in ALLOWED_EXTRA:
                    errors.append(f"unlisted file: {rel}")
        if errors:
            _fail(errors, warnings)
        return {
            "verified": True,
            "product": manifest.get("product"),
            "version": manifest.get("version"),
            "platform": manifest.get("platform"),
            "files": len(manifest["files"]),
            "anchor": str(anchor),
            "warnings": warnings,
        }


def _fail(errors: list[str], warnings: list[str]) -> None:
    print("RELEASE VERIFICATION FAILED:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    for warning in warnings:
        print(f"  WARN: {warning}", file=sys.stderr)
    raise SystemExit(2)


def dispatch_release_argv(argv: list[str]) -> int | None:
    if argv[:2] != ["release", "verify"]:
        return None
    rest = argv[2:]
    if not rest:
        print("用法: rosclaw release verify <bundle-dir|bundle.tar.gz> "
              "[--trusted-key PATH] [--trusted-fingerprint SHA256]", file=sys.stderr)
        return 2
    bundle = Path(rest[0])
    trusted_key = None
    fingerprint = ""
    idx = 1
    while idx < len(rest):
        if rest[idx] == "--trusted-key" and idx + 1 < len(rest):
            trusted_key = Path(rest[idx + 1])
            idx += 2
        elif rest[idx] == "--trusted-fingerprint" and idx + 1 < len(rest):
            fingerprint = rest[idx + 1]
            idx += 2
        else:
            print(f"未知参数 {rest[idx]}", file=sys.stderr)
            return 2
    if not bundle.exists():
        print(f"bundle 不存在: {bundle}", file=sys.stderr)
        return 2
    report = verify_bundle(
        bundle, trusted_key=trusted_key, trusted_fingerprint=fingerprint
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
