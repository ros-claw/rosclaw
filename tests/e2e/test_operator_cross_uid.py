"""T1：四 UID、四真实进程的 operator 安全 e2e（docker 容器内运行）。

本测试**不伪造 peer、不同 UID 绕行**——容器内创建
rosclawd/agent/operator/worker 四个真实用户与四个真实进程，
正向唯一成功、负向全部 fail closed。

无 docker 或镜像构建不可用（离线）时跳过并如实说明；CI 的
`cross-uid-operator-e2e` job 负责构建镜像并必跑。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SH = REPO / "scripts" / "e2e" / "operator_cross_uid" / "run.sh"
IMAGE = "rosclaw-operator-cross-uid-e2e:local"


def _docker_ready() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker CLI unavailable"
    try:
        probe = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker daemon unavailable: {exc}"
    if probe.returncode != 0:
        return False, f"docker daemon unavailable: {probe.stderr.strip()[:120]}"
    return True, ""


@pytest.mark.integration
def test_cross_uid_operator_e2e() -> None:
    ready, reason = _docker_ready()
    if not ready:
        pytest.skip(f"T1 cross-UID e2e requires docker: {reason}")
    result = subprocess.run(
        ["bash", str(RUN_SH)], capture_output=True, text=True, timeout=1200
    )
    output = result.stdout + result.stderr
    if result.returncode == 127 or "error during connect" in output.lower():
        pytest.skip(f"docker run unavailable: {output[-300:]}")
    if "Cannot connect to the Docker daemon" in output or "registry" in output.lower() and result.returncode != 0:
        pytest.skip(f"image build/run unavailable (offline?): {output[-300:]}")
    assert result.returncode == 0, f"T1 cross-UID e2e failed:\n{output[-3000:]}"
    assert "T1 RESULT" in result.stdout
