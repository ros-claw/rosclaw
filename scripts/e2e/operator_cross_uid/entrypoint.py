#!/usr/bin/env python3
"""T1 跨 UID 四进程 operator e2e（容器内驱动，root 编排）。

角色（真实不同 UID、真实独立进程、真实 socket IPC）：

* rcd  (uid 2001, rosclawd-test)     — rosclawd + enrollment 管理员
* rca  (uid 2002, rosclaw-agent-test)— agentd 侧：创建 proposal
* rco  (uid 2003, rosclaw-operator)  — operatord 侧：enroll/sign/decide
* rcw  (uid 2004, rosclaw-worker)    — worker：对所有控制 socket 必须 EACCES

正向唯一成功；负向全部 fail closed。退出码 0 = 全部断言通过。
"""

from __future__ import annotations

import grp
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/tmp/e2e")
REPO = Path(os.environ.get("ROSCLAW_REPO", "/repo"))
PYTHON = sys.executable

USERS = {
    "rcd": 2001,
    "rca": 2002,
    "rco": 2003,
    "rcw": 2004,
}
CONTROL_GROUP = "rccontrol"

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def sh(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=120, **kwargs)


def as_user(user: str, script: str, *args: str, extra_groups: list[str] | None = None,
            env: dict | None = None, background: bool = False):
    """以指定用户运行 driver 子命令。"""
    uid = USERS[user]
    cmd = [
        "setpriv",
        f"--reuid={uid}",
        f"--regid={uid}",
        *(["--groups", ",".join(extra_groups)] if extra_groups else ["--clear-groups"]),
        PYTHON,
        str(REPO / "scripts" / "e2e" / "operator_cross_uid" / "driver.py"),
        script,
        *args,
    ]
    full_env = dict(os.environ)
    full_env.update(env or {})
    if background:
        return subprocess.Popen(cmd, env=full_env)
    return sh(cmd, env=full_env)


def setup_users() -> None:
    sh(["groupadd", "-f", CONTROL_GROUP])
    for name, uid in USERS.items():
        sh(["useradd", "-u", str(uid), "-M", "-s", "/usr/sbin/nologin", name])
    # 控制组：daemon+agent+operator；worker 绝不入组。
    sh(["usermod", "-aG", CONTROL_GROUP, "rcd"])
    sh(["usermod", "-aG", CONTROL_GROUP, "rca"])
    sh(["usermod", "-aG", CONTROL_GROUP, "rco"])
    shutil.rmtree(ROOT, ignore_errors=True)
    for name in USERS:
        home = ROOT / name
        home.mkdir(parents=True)
        shutil.chown(home, user=name, group=name)
        os.chmod(home, 0o700)
    # daemon runtime 目录在 0700 home 之外：rcd:rccontrol 0750
    # （P0-2 布局——跨 UID 可达，worker 不可达）。
    run_dir = ROOT / "runtime" / "rosclawd"
    run_dir.mkdir(parents=True)
    shutil.chown(run_dir, user="rcd", group=CONTROL_GROUP)
    os.chmod(run_dir, 0o750)
    # shared：operator 导出公钥给管理员（组可写）。
    shared = ROOT / "shared"
    shared.mkdir()
    shutil.chown(shared, user="root", group=CONTROL_GROUP)
    os.chmod(shared, 0o775)


def main() -> int:
    setup_users()
    env = {"PYTHONPATH": str(REPO)}
    # 1. 启动 rosclawd（uid rcd，控制组 socket）。
    daemon_proc = as_user(
        "rcd", "daemon", extra_groups=[CONTROL_GROUP], env=env, background=True
    )
    try:
        sock = ROOT / "runtime" / "rosclawd" / "rosclawd.sock"
        for _ in range(100):
            if sock.exists():
                break
            if daemon_proc.poll() is not None:
                record("daemon_starts", False, f"exit={daemon_proc.returncode}")
                return finish(1)
            time.sleep(0.1)
        record("daemon_starts", sock.exists())
        # socket 权限：0660 + rccontrol。
        st = sock.stat()
        record(
            "daemon_socket_0660_group",
            (st.st_mode & 0o777) == 0o660 and st.st_gid == grp.getgrnam(CONTROL_GROUP).gr_gid,
            oct(st.st_mode & 0o777),
        )

        # 2. worker 对所有控制 socket EACCES。
        r = as_user("rcw", "probe_daemon_socket", env=env)
        record("worker_daemon_socket_eacces", "EACCES" in r.stdout, r.stdout.strip()[-80:])

        # 3. operatord 侧 enroll（uid rco，私钥 0600 于 rco 私有 home）。
        r = as_user("rco", "enroll", extra_groups=[CONTROL_GROUP], env=env)
        record("operator_enroll", r.returncode == 0 and "enrollment_id" in r.stdout,
               r.stderr.strip()[-120:])
        # worker 读 operator 私钥 → EACCES（0700 私有 home）。
        r = as_user("rcw", "probe_operator_key", env=env)
        record("worker_operator_key_eacces", "EACCES" in r.stdout, r.stdout.strip()[-80:])

        # 4. agent 试图抢注 enrollment（非管理员）→ PERMISSION_DENIED。
        r = as_user("rca", "register_attacker", extra_groups=[CONTROL_GROUP], env=env)
        record("agent_cannot_register_enrollment", "PERMISSION_DENIED" in r.stdout,
               r.stdout.strip()[-120:])

        # 5. daemon 管理员（rcd）登记 operator 公钥。
        r = as_user("rcd", "register_operator", extra_groups=[CONTROL_GROUP], env=env)
        record("admin_registers_operator", r.returncode == 0 and '"registered": true' in r.stdout,
               (r.stdout + r.stderr).strip()[-160:])

        # 6. agent 创建 proposal（uid rca，控制组成员）。
        r = as_user("rca", "create_proposal", extra_groups=[CONTROL_GROUP], env=env)
        record("agent_creates_proposal", r.returncode == 0 and "request_id" in r.stdout,
               (r.stdout + r.stderr).strip()[-160:])
        proposal_id = ""
        if r.returncode == 0:
            import contextlib

            with contextlib.suppress(json.JSONDecodeError):
                proposal_id = json.loads(r.stdout)["request_id"]

        # 7. agent 直接 decide（自签 key）→ PERMISSION_DENIED。
        if proposal_id:
            r = as_user("rca", "agent_self_decide", proposal_id,
                        extra_groups=[CONTROL_GROUP], env=env)
            record(
                "agent_cannot_decide",
                "PERMISSION_DENIED" in r.stdout or "DENIED_AT_CHALLENGE" in r.stdout,
                r.stdout.strip()[-120:],
            )

        # 8. operator 走完整协议：challenge → sign → decide → receipt 验证。
        if proposal_id:
            r = as_user("rco", "operator_decide", proposal_id,
                        extra_groups=[CONTROL_GROUP], env=env)
            record("operator_full_decision_chain", r.returncode == 0 and "RECEIPT_OK" in r.stdout,
                   (r.stdout + r.stderr).strip()[-200:])

        # 9. agent 轮询终态 receipt（SHADOW，actuated=false）。
        if proposal_id:
            r = as_user("rca", "await_action", extra_groups=[CONTROL_GROUP], env=env)
            record("action_terminal_shadow", r.returncode == 0 and "TERMINAL_OK" in r.stdout,
                   (r.stdout + r.stderr).strip()[-200:])

        # 10. 内部方法经 IPC 直调 → METHOD_NOT_ALLOWED；arm 非管理员 → 拒。
        r = as_user("rca", "probe_internal_methods", extra_groups=[CONTROL_GROUP], env=env)
        record("internal_arm_permit_not_ipc", "INTERNAL_OK" in r.stdout,
               r.stdout.strip()[-160:])

        # 11. daemon 重启：enrollment 持久化 + nonce 焚毁持久化（重放拒）。
        daemon_proc.terminate()
        daemon_proc.wait(timeout=15)
        daemon_proc = as_user(
            "rcd", "daemon", extra_groups=[CONTROL_GROUP], env=env, background=True
        )
        for _ in range(100):
            if sock.exists():
                break
            time.sleep(0.1)
        r = as_user("rcd", "check_registry_after_restart", extra_groups=[CONTROL_GROUP], env=env)
        record("enrollment_survives_restart", "REGISTRY_OK" in r.stdout,
               (r.stdout + r.stderr).strip()[-160:])
    finally:
        daemon_proc.terminate()
        try:
            daemon_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon_proc.kill()
    return finish(0)


def finish(code: int) -> int:
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n== T1 RESULT: {len(RESULTS) - len(failed)}/{len(RESULTS)} passed ==", flush=True)
    if failed:
        print("FAILED: " + ", ".join(failed), flush=True)
        return 1
    return code


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("entrypoint must run as root (user creation)", file=sys.stderr)
        sys.exit(2)
    sys.exit(main())
