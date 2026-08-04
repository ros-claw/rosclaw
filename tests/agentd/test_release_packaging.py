"""PR-12 发布打包测试：构建脚本产出结构完整、安装/回滚脚本语义正确。

- bundle 含 src/pyproject/packages dist/third_party/manifest/install/rollback
- manifest hash 与文件内容一致
- install.sh 原子切换 current/previous；rollback.sh 校验 manifest 后回切
（在临时目录用替身 venv/pip 做全dry 结构验证，不做真实安装）
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "scripts" / "build_release.sh"
INSTALL = REPO / "scripts" / "release" / "install_release.sh"
ROLLBACK = REPO / "scripts" / "release" / "rollback.sh"


@pytest.mark.slow
def test_build_release_bundle_structure(tmp_path: Path) -> None:
    env = dict(os.environ)
    result = subprocess.run(
        ["bash", str(BUILD)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    dist = REPO / "dist"
    bundles = list(dist.glob("rosclaw-*-linux-*.tar.gz"))
    assert bundles, "no bundle produced"
    bundle = bundles[0]
    with tarfile.open(bundle) as tf:
        names = tf.getnames()
        root = names[0].split("/")[0]
        expected = [
            f"{root}/pyproject.toml",
            f"{root}/manifest.json",
            f"{root}/install.sh",
            f"{root}/rollback.sh",
            f"{root}/packages/rosclaw-tui/package.json",
            f"{root}/packages/rosclaw-tui/package-lock.json",
            f"{root}/packages/rosclaw-modeld/package.json",
            f"{root}/packages/rosclaw-modeld/package-lock.json",
            f"{root}/third_party/pi/LICENSE",
            f"{root}/third_party/pi/NOTICE.md",
        ]
        for needle in expected:
            assert needle in names, f"missing {needle}"
        # TUI/modeld 已构建产物必须进包（安装侧可离线 npm ci 重建）。
        assert any("rosclaw-tui/dist/src/main.js" in n for n in names)
        assert any("rosclaw-modeld/dist/src/main.js" in n for n in names)
        manifest = json.loads(tf.extractfile(f"{root}/manifest.json").read())
        assert manifest["product"] == "rosclaw"
        assert manifest["files"], "manifest must hash every file"


class TestInstallRollbackSemantics:
    def test_scripts_syntax_and_guards(self) -> None:
        for script in (BUILD, INSTALL, ROLLBACK):
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True)
            assert result.returncode == 0, f"{script} syntax error"

    def test_rollback_requires_previous(self, tmp_path: Path) -> None:
        prefix = tmp_path / "prefix"
        (prefix / "current").mkdir(parents=True)
        env = dict(os.environ, ROSCLAW_PREFIX=str(prefix))
        result = subprocess.run(
            ["bash", str(ROLLBACK)], capture_output=True, text=True, env=env
        )
        assert result.returncode == 2
        assert "没有可回滚" in result.stderr

    def test_rollback_requires_manifest(self, tmp_path: Path) -> None:
        prefix = tmp_path / "prefix"
        (prefix / "current").mkdir(parents=True)
        (prefix / "previous").mkdir()
        env = dict(os.environ, ROSCLAW_PREFIX=str(prefix))
        result = subprocess.run(
            ["bash", str(ROLLBACK)], capture_output=True, text=True, env=env
        )
        assert result.returncode == 2
        assert "manifest" in result.stderr

    def test_rollback_swaps_current(self, tmp_path: Path) -> None:
        prefix = tmp_path / "prefix"
        (prefix / "current").mkdir(parents=True)
        (prefix / "current" / "marker_new").write_text("new")
        # R6：rollback 现在重验签名+hash——previous 必须是合法签名包。
        _make_mini_bundle(prefix / "previous", tmp_path / "signing")
        (prefix / "previous" / "marker_old").write_text("old")
        env = dict(os.environ, ROSCLAW_PREFIX=str(prefix),
                   ROSCLAW_RELEASE_KEY=str(tmp_path / "signing" / "pub.pem"))
        result = subprocess.run(
            ["bash", str(ROLLBACK)], capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, result.stderr
        assert (prefix / "current" / "marker_old").exists()
        assert list(prefix.glob("failed-*")), "failed version must be preserved"


@pytest.mark.slow
class TestSignedBundle:
    def test_build_produces_signed_manifest_and_offline_assets(self, tmp_path: Path) -> None:
        import subprocess
        import tarfile

        env = dict(os.environ, ROSCLAW_SIGNING_HOME=str(tmp_path / "signing"))
        result = subprocess.run(
            ["bash", str(REPO / "scripts" / "build_release.sh")],
            cwd=REPO, capture_output=True, text=True, timeout=900, env=env,
        )
        assert result.returncode == 0, result.stderr[-1500:]
        bundles = sorted((REPO / "dist").glob("rosclaw-*-linux-*.tar.gz"))
        bundle = bundles[-1]
        stage = tmp_path / "stage"
        stage.mkdir()
        with tarfile.open(bundle) as tf:
            tf.extractall(stage)
        root = next(stage.iterdir())
        # 签名与公钥存在，SBOM（CycloneDX + 快照）与离线资产存在。
        for name in ("manifest.json", "manifest.sig", "bundle-signing-public.pem",
                     "sbom-python.txt", "sbom-rosclaw-tui.txt", "sbom-rosclaw-modeld.txt",
                     "sbom-cyclonedx.json"):
            assert (root / name).exists(), name
        sbom = json.loads((root / "sbom-cyclonedx.json").read_text())
        assert sbom["bomFormat"] == "CycloneDX" and sbom["components"]
        assert (root / "vendor" / "node_modules_pack" / "rosclaw-tui.tar.gz").exists()
        assert (root / "vendor" / "node_modules_pack" / "rosclaw-modeld.tar.gz").exists()
        # 签名有效。
        verify = subprocess.run(
            ["openssl", "dgst", "-sha256", "-verify",
             str(root / "bundle-signing-public.pem"),
             "-signature", str(root / "manifest.sig"),
             str(root / "manifest.json")],
            capture_output=True, text=True,
        )
        assert verify.returncode == 0, verify.stderr
        # 篡改 manifest 中任一文件 → 验签/has 必拒（这里改一个源码文件）。
        target = next((root / "src" / "rosclaw").rglob("*.py"))
        target.write_text("tampered")
        verify2 = subprocess.run(
            ["openssl", "dgst", "-sha256", "-verify",
             str(root / "bundle-signing-public.pem"),
             "-signature", str(root / "manifest.sig"),
             str(root / "manifest.json")],
            capture_output=True, text=True,
        )
        # manifest.json 本身未变 → 签名仍有效，但文件 hash 校验必败。
        assert verify2.returncode == 0
        check = subprocess.run(
            ["python3", "-c", _HASH_CHECK_SNIPPET, str(root)],
            capture_output=True, text=True,
        )
        assert check.returncode == 2
        assert "tampered" in check.stderr
        # R6：无包外信任锚 → 安装器直接拒绝（包内公钥不能自证）。
        no_anchor = subprocess.run(
            ["bash", str(root / "install.sh"), "--offline"],
            capture_output=True, text=True, timeout=120,
            env=dict(os.environ, ROSCLAW_PREFIX=str(tmp_path / "prefix0"),
                     ROSCLAW_RELEASE_KEY=str(tmp_path / "nonexistent.pem")),
        )
        assert no_anchor.returncode == 2
        assert "信任锚" in no_anchor.stderr
        # 安装器在篡改下直接拒绝（verify-before-execute；锚来自包外）。
        install = subprocess.run(
            ["bash", str(root / "install.sh"), "--offline",
             "--trusted-key", str(tmp_path / "signing" / "dev-signing-public.pem")],
            capture_output=True, text=True, timeout=120,
            env=dict(os.environ, ROSCLAW_PREFIX=str(tmp_path / "prefix")),
        )
        assert install.returncode == 2
        assert "篡改" in install.stderr or "tampered" in install.stderr


_HASH_CHECK_SNIPPET = (
    "import hashlib, json, sys\n"
    "from pathlib import Path\n"
    "root = Path(sys.argv[1])\n"
    "manifest = json.loads((root / 'manifest.json').read_text())\n"
    "bad = []\n"
    "for rel, expected in manifest['files'].items():\n"
    "    path = root / rel\n"
    "    if not path.exists():\n"
    "        bad.append(f'missing: {rel}')\n"
    "        continue\n"
    "    actual = hashlib.sha256(path.read_bytes()).hexdigest()\n"
    "    if actual != expected:\n"
    "        bad.append(f'tampered: {rel}')\n"
    "if bad:\n"
    "    print('\\n'.join(bad), file=sys.stderr)\n"
    "    sys.exit(2)\n"
)


# ----------------------------------------------------------------------
# T5（二次复核 R6）：发布供应链攻击矩阵。
# ----------------------------------------------------------------------


def _make_mini_bundle(root: Path, signing: Path) -> None:
    """构造最小签名 bundle（真实 openssl ECDSA）。"""
    import hashlib

    signing.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout",
         "-out", str(signing / "priv.pem")],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["openssl", "ec", "-in", str(signing / "priv.pem"), "-pubout",
         "-out", str(signing / "pub.pem")],
        check=True, capture_output=True,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("payload-a")
    (root / "src").mkdir()
    (root / "src" / "x.py").write_text("print('x')\n")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"product": "t", "version": "0", "files": files}))
    (root / "bundle-signing-public.pem").write_bytes((signing / "pub.pem").read_bytes())
    subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", str(signing / "priv.pem"),
         "-out", str(root / "manifest.sig"), str(root / "manifest.json")],
        check=True, capture_output=True,
    )


class TestReleaseTrustAnchor:
    """T5：信任锚在包外；篡改/替换/夹带全部拒绝。"""

    def test_valid_bundle_verifies_with_anchor(self, tmp_path: Path) -> None:
        from rosclaw.release_verify import verify_bundle

        root, signing = tmp_path / "bundle", tmp_path / "signing"
        _make_mini_bundle(root, signing)
        report = verify_bundle(root, trusted_key=signing / "pub.pem")
        assert report["verified"] and report["files"] == 2

    def test_no_anchor_refuses(self, tmp_path: Path, monkeypatch) -> None:
        from rosclaw.release_verify import verify_bundle

        root, signing = tmp_path / "bundle", tmp_path / "signing"
        _make_mini_bundle(root, signing)
        monkeypatch.setenv("ROSCLAW_RELEASE_KEY", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "nohome")
        with pytest.raises(SystemExit) as exit_info:
            verify_bundle(root)
        assert exit_info.value.code == 2

    def test_payload_tamper_refused(self, tmp_path: Path) -> None:
        from rosclaw.release_verify import verify_bundle

        root, signing = tmp_path / "bundle", tmp_path / "signing"
        _make_mini_bundle(root, signing)
        (root / "a.txt").write_text("tampered-one-byte")
        with pytest.raises(SystemExit) as exit_info:
            verify_bundle(root, trusted_key=signing / "pub.pem")
        assert exit_info.value.code == 2

    def test_manifest_and_key_replacement_refused(self, tmp_path: Path) -> None:
        """攻击者同时替换 manifest、签名与包内公钥——包外锚仍拒绝。"""
        from rosclaw.release_verify import verify_bundle

        root, signing = tmp_path / "bundle", tmp_path / "signing"
        _make_mini_bundle(root, signing)
        attacker = tmp_path / "attacker"
        _make_mini_bundle(attacker, tmp_path / "attacker-signing")
        # 攻击者重打包：换成自己的 manifest+签名+公钥。
        (root / "manifest.json").write_bytes((attacker / "manifest.json").read_bytes())
        (root / "manifest.sig").write_bytes((attacker / "manifest.sig").read_bytes())
        (root / "bundle-signing-public.pem").write_bytes(
            (attacker / "bundle-signing-public.pem").read_bytes()
        )
        with pytest.raises(SystemExit) as exit_info:
            verify_bundle(root, trusted_key=signing / "pub.pem")
        assert exit_info.value.code == 2

    def test_extra_unlisted_file_refused(self, tmp_path: Path) -> None:
        from rosclaw.release_verify import verify_bundle

        root, signing = tmp_path / "bundle", tmp_path / "signing"
        _make_mini_bundle(root, signing)
        (root / "evil.sh").write_text("#!/bin/sh\nid\n")
        with pytest.raises(SystemExit) as exit_info:
            verify_bundle(root, trusted_key=signing / "pub.pem")
        assert exit_info.value.code == 2

    def test_symlink_refused(self, tmp_path: Path) -> None:
        from rosclaw.release_verify import verify_bundle

        root, signing = tmp_path / "bundle", tmp_path / "signing"
        _make_mini_bundle(root, signing)
        (root / "link").symlink_to("/etc/passwd")
        with pytest.raises(SystemExit) as exit_info:
            verify_bundle(root, trusted_key=signing / "pub.pem")
        assert exit_info.value.code == 2

    def test_fingerprint_pinning(self, tmp_path: Path) -> None:
        import hashlib

        from rosclaw.release_verify import verify_bundle

        root, signing = tmp_path / "bundle", tmp_path / "signing"
        _make_mini_bundle(root, signing)
        proc = subprocess.run(
            ["openssl", "pkey", "-pubin", "-in", str(signing / "pub.pem"), "-outform", "DER"],
            capture_output=True, check=True,
        )
        good_fp = hashlib.sha256(proc.stdout).hexdigest()
        report = verify_bundle(
            root, trusted_key=signing / "pub.pem", trusted_fingerprint=good_fp
        )
        assert report["verified"]
        with pytest.raises(SystemExit) as exit_info:
            verify_bundle(root, trusted_key=signing / "pub.pem", trusted_fingerprint="0" * 64)
        assert exit_info.value.code == 2

    def test_rollback_refuses_tampered_previous(self, tmp_path: Path) -> None:
        """R6：被篡改的 previous 不得成为回滚目标。"""
        prefix = tmp_path / "prefix"
        previous = prefix / "previous"
        _make_mini_bundle(previous, tmp_path / "signing")
        current = prefix / "current"
        current.mkdir(parents=True)
        # 篡改 previous 的一个 payload 文件。
        (previous / "a.txt").write_text("tampered")
        result = subprocess.run(
            ["bash", str(ROLLBACK)],
            capture_output=True, text=True,
            env=dict(os.environ, ROSCLAW_PREFIX=str(prefix),
                     ROSCLAW_RELEASE_KEY=str(tmp_path / "signing" / "pub.pem")),
        )
        assert result.returncode == 2
        assert "篡改" in result.stderr or "tampered" in result.stderr
        assert current.exists(), "拒绝回滚不得改变 current"
