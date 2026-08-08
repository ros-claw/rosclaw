"""真实前台终端 human confirmation（二次复核 P0-1/T2）。

初版的 ``_human_present()`` 只是"能打开 /dev/tty"——证明的是 tty
存在，不是"人确认了本次动作"。本模块实现最低要求：

1. 在可信终端（/dev/tty，非 stdin——防 ``echo y |`` 重定向伪造）显示
   不可变动作卡（含 display_hash / challenge nonce 片段）；
2. 读取显式 ``Y``/``N``；默认、超时、EOF 一律 deny；
3. 用 SO_PEERCRED 拿到的请求方 PID 校验其进程组是对应 TTY 的
   **前台进程组**（/proc/<pid>/stat 的 tpgid == pgrp）——后台进程
   调用 socket 不得触发批准；
4. 确认结果绑定 challenge（proposal_id + nonce + display_hash +
   decision + expires_at），由 proof 签名覆盖。
"""

from __future__ import annotations

import os
import select
import stat
from dataclasses import dataclass

CONFIRM_TIMEOUT_SEC = 60.0


@dataclass(frozen=True)
class HumanPromptResult:
    """确认结果。decision: True=Y / False=N；None=无法确认（超时/EOF/无 tty）。"""

    decision: bool | None
    method: str
    detail: str


def requester_is_foreground(pid: int) -> bool:
    """请求方进程必须位于其控制终端的前台进程组。

    /proc/<pid>/stat 字段（1-indexed）：2=comm(可含空格与括号) 3=state
    4=ppid 5=pgrp 6=session 7=tty_nr 8=tpgid。
    comm 用最后一个 ')' 定位，避免空格/括号注入。
    """
    if pid <= 0:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            stat_line = handle.read()
    except OSError:
        return False
    rparen = stat_line.rfind(")")
    if rparen < 0:
        return False
    fields = stat_line[rparen + 2 :].split()
    # fields[0]=state(3) → fields[2]=pgrp(5), fields[4]=tty_nr(7), fields[5]=tpgid(8)
    if len(fields) < 6:
        return False
    try:
        pgrp = int(fields[2])
        tty_nr = int(fields[4])
        tpgid = int(fields[5])
    except ValueError:
        return False
    if tty_nr == 0 or tpgid <= 0:
        return False
    return tpgid == pgrp


def render_card(
    *,
    title: str,
    summary: str,
    risk_tier: str,
    mode: str,
    capability: str,
    parameters: dict,
    display_hash: str,
    challenge_nonce: str,
    expires_at: str,
) -> str:
    lines = [
        "",
        "┌─ ROSClaw OPERATOR DECISION ─────────────────────────",
        f"│ 动作:     {title}",
        f"│ 说明:     {summary}",
        f"│ 风险:     {risk_tier}    模式: {mode}",
        f"│ 能力:     {capability}",
    ]
    for key, value in sorted(parameters.items()):
        lines.append(f"│ 参数 {key} = {value}")
    lines += [
        f"│ display_hash: {display_hash}",
        f"│ challenge:    {challenge_nonce[:16]}…",
        f"│ 过期:         {expires_at}",
        "└─────────────────────────────────────────────────────",
        "输入 Y 批准 / N 拒绝（默认拒绝，60s 超时拒绝）: ",
    ]
    return "\n".join(lines)


def confirm_on_tty(card: str, *, timeout_sec: float = CONFIRM_TIMEOUT_SEC) -> HumanPromptResult:
    """在 /dev/tty 显示卡片并读取显式 Y/N。

    任何不可用（无 tty、EOF、超时、非法输入后 EOF）都是 deny——
    绝不回退为自动批准。
    """
    try:
        tty_fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError as exc:
        return HumanPromptResult(
            None, "tty-yn", f"no controlling terminal ({exc}) — decision refused"
        )
    try:
        return _confirm_on_fd(tty_fd, card, timeout_sec=timeout_sec, method="tty-yn")
    finally:
        os.close(tty_fd)


def confirm_on_requester_tty(
    pid: int, card: str, *, timeout_sec: float = CONFIRM_TIMEOUT_SEC
) -> HumanPromptResult:
    """Display and read confirmation on the foreground requester's terminal.

    operatord is intentionally a separate long-lived process and may have no
    controlling terminal of its own.  The socket peer PID is authenticated via
    ``SO_PEERCRED`` and checked separately to be in its terminal's foreground
    process group; opening that peer's fd 0 keeps the prompt on the actual chat
    surface without trusting redirected stdin or weakening the Y/N requirement.
    """

    if pid <= 0:
        return HumanPromptResult(None, "requester-tty-yn", "invalid requester pid")
    path = f"/proc/{pid}/fd/0"
    try:
        # Opening another process' procfs fd can be denied by Yama even when
        # both processes have the same uid.  Resolving the fd symlink and then
        # opening the actual PTY preserves the same terminal boundary.
        tty_path = os.readlink(path)
        if not tty_path.startswith("/dev/"):
            raise OSError(f"requester fd 0 does not resolve to a device: {tty_path}")
        tty_fd = os.open(tty_path, os.O_RDWR | os.O_NOCTTY)
        opened = os.fstat(tty_fd)
        if not stat.S_ISCHR(opened.st_mode) or os.ttyname(tty_fd) != tty_path:
            os.close(tty_fd)
            raise OSError("requester terminal changed while resolving fd 0")
    except OSError as exc:
        return HumanPromptResult(
            None,
            "requester-tty-yn",
            f"requester terminal unavailable ({exc}) — decision refused",
        )
    try:
        if not os.isatty(tty_fd):
            return HumanPromptResult(
                None, "requester-tty-yn", "requester stdin is not a tty — decision refused"
            )
        return _confirm_on_fd(
            tty_fd, card, timeout_sec=timeout_sec, method="requester-tty-yn"
        )
    finally:
        os.close(tty_fd)


def _confirm_on_fd(
    tty_fd: int,
    card: str,
    *,
    timeout_sec: float,
    method: str,
) -> HumanPromptResult:
    try:
        os.write(tty_fd, (card + "\n").encode())
        ready, _, _ = select.select([tty_fd], [], [], timeout_sec)
        if not ready:
            os.write(tty_fd, b"\n[timeout - treated as DENY]\n")
            return HumanPromptResult(None, method, "confirmation timed out — denied")
        data = os.read(tty_fd, 64)
        if not data:
            os.write(tty_fd, b"\n[EOF - treated as DENY]\n")
            return HumanPromptResult(None, method, "tty EOF — denied")
        answer = data.decode(errors="replace").strip().upper()
        os.write(tty_fd, b"\n")
        if answer.startswith("Y"):
            return HumanPromptResult(True, method, "operator approved on foreground tty")
        return HumanPromptResult(False, method, f"operator denied ({answer or 'empty'})")
    except OSError as exc:
        return HumanPromptResult(None, method, f"tty I/O failed ({exc}) — decision refused")
