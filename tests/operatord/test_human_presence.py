"""T2：human-presence 真实 PTY 测试（二次复核 P0-1）。

- 无 /dev/tty（CI/重定向）：不得批准；
- PTY 前台输入 Y → 只批准本次；输入 N/EOF/超时 → deny；
- 请求方进程组校验：前台 pgrp==tpgid 才通过。
"""

from __future__ import annotations

import os
import pty
import select
import threading
import time

import pytest

from rosclaw.operatord.human import (
    HumanPromptResult,
    confirm_on_requester_tty,
    confirm_on_tty,
    render_card,
    requester_is_foreground,
)

CARD = render_card(
    title="播放提示音",
    summary="660Hz 0.6s 18%",
    risk_tier="LOW",
    mode="REAL",
    capability="limo.speaker.play_tone",
    parameters={"frequency_hz": 660},
    display_hash="abcd1234abcd1234",
    challenge_nonce="nonce-example-1234",
    expires_at="2026-08-04T20:00:00Z",
)


def _run_confirm_in_pty(answer: bytes | None, *, delay: float = 0.3) -> HumanPromptResult:
    """在真实 PTY（子进程新会话 + 控制终端）里执行 confirm_on_tty。

    父进程经 master 写入回答；子进程的结果经 pipe 传回。
    """
    master, slave = pty.openpty()
    result_r, result_w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(master)
            os.close(result_r)
            os.setsid()
            import fcntl
            import termios

            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            os.dup2(slave, 0)
            os.dup2(slave, 1)
            os.dup2(slave, 2)
            result = confirm_on_tty(CARD, timeout_sec=2.0)
            payload = f"{result.decision}|{result.method}".encode()
            os.write(result_w, payload)
            os._exit(0)
        except Exception:  # noqa: BLE001
            os.write(result_w, b"EXC|")
            os._exit(1)
    os.close(slave)
    os.close(result_w)
    try:
        if answer is not None:
            time.sleep(delay)
            os.write(master, answer)
        deadline = time.monotonic() + 8.0
        data = b""
        while time.monotonic() < deadline:
            ready, _, _ = select.select([result_r], [], [], 0.2)
            if ready:
                chunk = os.read(result_r, 64)
                if not chunk:
                    break
                data += chunk
                if b"|" in data:
                    break
        _, status = os.waitpid(pid, 0)
        text = data.decode(errors="replace")
        decision_s, _, method = text.partition("|")
        decision = {"True": True, "False": False, "None": None, "EXC": "EXC"}.get(decision_s)
        assert decision != "EXC", "child raised"
        return HumanPromptResult(decision, method, "")
    finally:
        os.close(master)
        os.close(result_r)


class TestConfirmOnTty:
    def test_explicit_y_approves(self) -> None:
        result = _run_confirm_in_pty(b"Y\n")
        assert result.decision is True
        assert result.method == "tty-yn"

    def test_explicit_n_denies(self) -> None:
        result = _run_confirm_in_pty(b"n\n")
        assert result.decision is False

    def test_garbage_denies(self) -> None:
        result = _run_confirm_in_pty(b"maybe\n")
        assert result.decision is False

    def test_no_input_times_out_as_deny(self) -> None:
        # 子进程 timeout_sec=2.0；父进程不写任何东西。
        result = _run_confirm_in_pty(None)
        assert result.decision is None

    def test_no_controlling_tty_denies(self) -> None:
        """当前 pytest 进程无控制终端（CI/重定向场景）→ 不得批准。"""
        try:
            fd = os.open("/dev/tty", os.O_RDONLY)
        except OSError:
            result = confirm_on_tty(CARD, timeout_sec=0.1)
            assert result.decision is None
            assert "no controlling terminal" in result.detail
        else:
            os.close(fd)
            pytest.skip("interactive session has a controlling tty")


def test_background_operatord_can_confirm_on_foreground_requester_tty(monkeypatch) -> None:
    master, slave = pty.openpty()
    tty_path = os.ttyname(slave)
    monkeypatch.setattr(os, "readlink", lambda _path: tty_path)
    try:
        def answer() -> None:
            time.sleep(0.2)
            os.write(master, b"Y\n")

        writer = threading.Thread(target=answer)
        writer.start()
        result = confirm_on_requester_tty(12345, CARD, timeout_sec=2.0)
        writer.join(timeout=2.0)
        assert result.decision is True
        assert result.method == "requester-tty-yn"
    finally:
        os.close(master)
        os.close(slave)


class TestForegroundCheck:
    def test_current_process_without_tty_not_foreground(self) -> None:
        try:
            fd = os.open("/dev/tty", os.O_RDONLY)
        except OSError:
            assert not requester_is_foreground(os.getpid())
        else:
            os.close(fd)
            pytest.skip("interactive session")

    def test_invalid_pid_not_foreground(self) -> None:
        assert not requester_is_foreground(0)
        assert not requester_is_foreground(-1)
        assert not requester_is_foreground(2**22)

    def test_pty_foreground_child_is_foreground(self) -> None:
        """真实 PTY 前台子进程：pgrp == tpgid → True。"""
        master, slave = pty.openpty()
        pid = os.fork()
        if pid == 0:
            os.close(master)
            os.setsid()
            import fcntl
            import termios

            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            os.dup2(slave, 0)
            ok = requester_is_foreground(os.getpid())
            os._exit(0 if ok else 1)
        os.close(slave)
        _, status = os.waitpid(pid, 0)
        os.close(master)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, (
            "PTY 前台进程应通过前台校验"
        )
