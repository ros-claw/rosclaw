"""T-SOAK（二次审计 Gate E）：安装产物 100 次启动/退出浸泡 + 资源趋势。

每轮：rosclaw chat --engine pi → 等品牌 header → /quit → 干净退出（rc=0）。
跟踪进程树 RSS 与 fd 数——前 10 轮均值 vs 后 10 轮均值，增长必须收敛
（无单调泄漏）。复用 test_product_journey 的构建/安装/PTY 设施。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tests.agentd.test_product_journey import (
    FakeModelServer,
    PtySession,
    _build_and_install,
)


def _tree_pids(root_pid: int) -> list[int]:
    """root_pid 的整棵子树（含自身），经 /proc ppid 链。"""
    children: dict[int, list[int]] = {}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            status = (pid_dir / "status").read_text()
            ppid = next(
                int(line.split()[1]) for line in status.splitlines() if line.startswith("PPid:")
            )
        except Exception:  # noqa: BLE001 - 进程随时消失
            continue
        children.setdefault(ppid, []).append(int(pid_dir.name))
    result: list[int] = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        result.append(pid)
        stack.extend(children.get(pid, []))
    return result


def _tree_rss_kb(root_pid: int) -> int:
    total = 0
    for pid in _tree_pids(root_pid):
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            total += next(
                int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:")
            )
        except Exception:  # noqa: BLE001
            continue
    return total


def _tree_fds(root_pid: int) -> int:
    total = 0
    for pid in _tree_pids(root_pid):
        try:
            total += len(list(Path(f"/proc/{pid}/fd").iterdir()))
        except Exception:  # noqa: BLE001
            continue
    return total


def _write_fake_home(home: Path, base_url: str) -> None:
    import json

    (home / "run").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "agent:\n  enabled: true\n  default_profile: embodied_default\n"
        "models:\n  backend: legacy\n  profiles:\n    embodied_default:\n"
        "      provider: kimi_code\n      model: fake-k3\n"
        f"      base_url: {base_url}\n"
        "      api_key_ref: env:FAKE_JOURNEY_KEY\n"
        "      capabilities: [llm.chat, llm.structured_decision, llm.tool_use]\n",
        encoding="utf-8",
    )
    (home / "agent").mkdir(parents=True, exist_ok=True)
    (home / "agent" / "settings.json").write_text(
        json.dumps({"defaultProvider": "journey-fake", "defaultModel": "fake-k3"}),
        encoding="utf-8",
    )
    (home / "agent" / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "journey-fake": {
                        "name": "Journey Fake",
                        "baseUrl": base_url,
                        "api": "openai-completions",
                        "apiKey": "$FAKE_JOURNEY_KEY",
                        "models": [
                            {
                                "id": "fake-k3",
                                "name": "Fake K3",
                                "contextWindow": 8192,
                                "maxTokens": 4096,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.slow
class TestStartExitSoak:
    ROUNDS = 100
    # 资源增长容忍：后 10 轮均值相对前 10 轮均值的上限。
    # （首轮有一次性缓存/映射 warmup，故比较均值而非首轮。）
    RSS_GROWTH_LIMIT = 1.30
    FD_GROWTH_LIMIT = 1.50

    def test_hundred_start_exit_cycles(self, tmp_path: Path) -> None:
        fake = FakeModelServer(log_path=tmp_path / "fake-requests.jsonl")
        prefix, _root = _build_and_install(tmp_path)
        home = tmp_path / "rh"
        _write_fake_home(home, fake.base_url)
        rosclaw = prefix / "bin" / "rosclaw"
        env = dict(
            os.environ,
            ROSCLAW_HOME=str(home),
            TERM="xterm",
            FAKE_JOURNEY_KEY="sk-fake-journey",
            KIMI_API_KEY="sk-fake-journey",
            PATH=f"{prefix / 'bin'}:{os.environ['PATH']}",
        )
        rss_samples: list[int] = []
        fd_samples: list[int] = []
        try:
            for round_no in range(self.ROUNDS):
                session = PtySession([str(rosclaw), "chat", "--engine", "pi"], env)
                try:
                    session.expect(b"ROSClaw Native Agent", timeout=90)
                    # 稳定后采样（header 出现 → 进程树已完全起来）。
                    time.sleep(0.5)
                    rss_samples.append(_tree_rss_kb(session.proc.pid))
                    fd_samples.append(_tree_fds(session.proc.pid))
                    session.send("/quit\r")
                    # 空会话不打印 resume 提示（journey 里有对话才有）——
                    # 浸泡断言干净退出本身。
                    session.proc.wait(timeout=30)
                    assert session.proc.returncode == 0, (
                        f"round {round_no}: 退出码 {session.proc.returncode}；"
                        f"尾部: {session.output[-300:]!r}"
                    )
                finally:
                    session.stop()
                # 每轮后确认进程树真的死了——泄漏的孤儿进程会污染后续采样。
                time.sleep(0.3)
                survivors = [
                    pid for pid in _tree_pids(session.proc.pid) if Path(f"/proc/{pid}").exists()
                ]
                assert not survivors, f"round {round_no}: 进程树残留 {survivors}"
        finally:
            fake.close()
        assert len(rss_samples) == self.ROUNDS
        head_rss = sum(rss_samples[:10]) / 10
        tail_rss = sum(rss_samples[-10:]) / 10
        head_fd = sum(fd_samples[:10]) / 10
        tail_fd = sum(fd_samples[-10:]) / 10
        (tmp_path / "soak-trend.txt").write_text(
            f"rss head10={head_rss:.0f}KB tail10={tail_rss:.0f}KB "
            f"ratio={tail_rss / head_rss:.3f}\n"
            f"fd  head10={head_fd:.0f} tail10={tail_fd:.0f} "
            f"ratio={tail_fd / head_fd:.3f}\n"
            f"rss series: {rss_samples}\nfd series: {fd_samples}\n",
            encoding="utf-8",
        )
        assert tail_rss <= head_rss * self.RSS_GROWTH_LIMIT, (
            f"RSS 单调增长：head10={head_rss:.0f}KB tail10={tail_rss:.0f}KB"
        )
        assert tail_fd <= head_fd * self.FD_GROWTH_LIMIT, (
            f"fd 单调增长：head10={head_fd:.0f} tail10={tail_fd:.0f}"
        )
