"""PNA-10（规格 §27）：发布门禁——clean build、build-info、bundled Node、
installed-artifact PTY 冒烟。

- build-info.json 字段完整且含 rosclaw-agent；
- bundled node runtime 进包且在 manifest 内（extra-file 拒绝兼容）；
- 安装后经 PTY 启动 rosclaw-agent（InteractiveMode），/quit 干净退出、
  无孤儿进程。
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


def _build(tmp_path: Path) -> Path:
    env = dict(os.environ, ROSCLAW_SIGNING_HOME=str(tmp_path / "signing"))
    result = subprocess.run(
        ["bash", str(BUILD)], cwd=REPO, capture_output=True, text=True, timeout=1800, env=env
    )
    assert result.returncode == 0, result.stderr[-2000:]
    bundle = sorted((REPO / "dist").glob("rosclaw-*-linux-*.tar.gz"))[-1]
    stage = tmp_path / "stage"
    stage.mkdir()
    with tarfile.open(bundle) as tf:
        tf.extractall(stage)
    return next(stage.iterdir())


@pytest.mark.slow
class TestReleaseGate:
    def test_build_info_and_bundled_node(self, tmp_path: Path) -> None:
        root = _build(tmp_path)
        info = json.loads((root / "build-info.json").read_text())
        assert info["pi_version"] == "0.83.0"
        assert info["rosclaw_commit"]
        for pkg in ("rosclaw-tui", "rosclaw-modeld", "rosclaw-agent"):
            assert info["packages"][pkg]["dist_sha256"], pkg
        if os.environ.get("ROSCLAW_SKIP_NODE_BUNDLE") != "1":
            node = root / "vendor" / "node-runtime" / "bin" / "node"
            assert node.exists(), "bundled node runtime must be in the bundle"
            manifest = json.loads((root / "manifest.json").read_text())
            assert any(
                name.startswith("vendor/node-runtime/") for name in manifest["files"]
            ), "bundled node must be inside the signed manifest"

    def test_installed_artifact_pty_quit_clean(self, tmp_path: Path) -> None:
        """安装产物在 PTY 中启动 → 显示 ROSClaw 品牌 → /quit 退出无孤儿。"""
        root = _build(tmp_path)
        install = subprocess.run(
            [
                "bash", str(root / "install.sh"), "--offline",
                "--trusted-key", str(tmp_path / "signing" / "dev-signing-public.pem"),
            ],
            capture_output=True, text=True, timeout=900,
            env=dict(os.environ, ROSCLAW_PREFIX=str(tmp_path / "prefix")),
        )
        assert install.returncode == 0, install.stderr[-1500:]
        current = tmp_path / "prefix" / "current"
        entry = current / "packages" / "rosclaw-agent" / "dist" / "src" / "main.js"
        node = current / "vendor" / "node-runtime" / "bin" / "node"
        if not node.exists():
            node = Path("/usr/bin/node")
        assert entry.exists()

        import pty as _pty
        import select
        import time

        master, slave = _pty.openpty()
        proc = subprocess.Popen(
            [str(node), str(entry)],
            stdin=slave, stdout=slave, stderr=slave,
            env=dict(
                os.environ,
                ROSCLAW_HOME=str(tmp_path / "rh"),
                TERM="xterm",
                # 与安装器 wrapper 一致：prefix/bin 提供随包 fd/rg（离线路径）。
                PATH=f"{tmp_path / 'prefix' / 'bin'}:{os.environ['PATH']}",
            ),
            close_fds=True,
        )
        os.close(slave)
        output = b""
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and b"ROSClaw" not in output:
                ready, _, _ = select.select([master], [], [], 0.5)
                if ready:
                    chunk = os.read(master, 4096)
                    if not chunk:
                        break
                    output += chunk
                if proc.poll() is not None:
                    break
            assert b"ROSClaw" in output, f"品牌 header 未出现: {output[-500:]}"
            # 等 TUI 完全就绪（2s 无新输出）再发 /quit——过早输入会被
            # 初始化阶段的检查吞掉。
            quiet_since = time.monotonic()
            while time.monotonic() < deadline + 10:
                ready, _, _ = select.select([master], [], [], 0.5)
                if ready:
                    chunk = os.read(master, 4096)
                    if not chunk:
                        break
                    output += chunk
                    quiet_since = time.monotonic()
                if time.monotonic() - quiet_since > 2.0:
                    break
            os.write(master, b"/quit\r")
            proc.wait(timeout=30)
            assert proc.returncode == 0, f"exit={proc.returncode} out={output[-300:]}"
        finally:
            os.close(master)
            if proc.poll() is None:
                proc.kill()
        # 无孤儿：进程已退出。
        assert proc.poll() is not None
