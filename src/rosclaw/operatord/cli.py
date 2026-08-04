"""`rosclaw operatord enroll|start|status` — operatord CLI（审计 P0-01）。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


def dispatch_operatord_argv(argv: list[str]) -> int | None:
    if argv[:1] != ["operatord"]:
        return None
    if len(argv) < 2:
        print("用法: rosclaw operatord <enroll|start|status> [--home DIR]", file=sys.stderr)
        return 2
    command = argv[1]
    home = Path(
        argv[argv.index("--home") + 1]
        if "--home" in argv
        else os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw")
    )

    if command == "enroll":
        from rosclaw.operatord.enrollment import EnrollmentError, enroll

        try:
            enrollment = enroll(home / "operatord")
        except EnrollmentError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "enrollment_id": enrollment.enrollment_id,
                    "fingerprint": enrollment.fingerprint,
                    "uid": enrollment.uid,
                    "note": "key 仅存于 operatord home（0600）；"
                    "daemon ACL 登记见 `rosclaw operatord register-daemon`。",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if command == "register-daemon":
        from rosclaw.daemon.client import DaemonClient, DaemonClientError
        from rosclaw.operatord.enrollment import EnrollmentError, load_enrollment

        daemon_socket = home / "run" / "rosclawd.sock"
        if not daemon_socket.exists():
            print(f"rosclawd socket 不存在：{daemon_socket}", file=sys.stderr)
            return 2
        try:
            enrollment = load_enrollment(home / "operatord")
        except EnrollmentError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        client = DaemonClient(socket_path=daemon_socket)
        try:
            result = client.register_operator_enrollment(
                enrollment.enrollment_id, enrollment.key.hex()
            )
        except DaemonClientError as exc:
            print(f"daemon 登记失败：{exc.code}: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if command == "start":
        import contextlib

        from rosclaw.operatord.enrollment import EnrollmentError
        from rosclaw.operatord.server import run_operatord

        async def _serve() -> None:
            daemon = await run_operatord(
                home=home,
                daemon_socket=home / "run" / "rosclawd.sock",
                require_human_presence="--no-human-presence-check" not in argv,
            )
            print(f"rosclaw-operatord listening on {daemon._path}")
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()

        try:
            asyncio.run(_serve())
        except EnrollmentError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            return 0
        return 0

    if command == "status":
        from rosclaw.operatord.enrollment import EnrollmentError, load_enrollment

        try:
            enrollment = load_enrollment(home / "operatord")
            enrolled = True
        except EnrollmentError:
            enrolled = False
            enrollment = None
        sock = home / "run" / "operatord.sock"
        print(
            json.dumps(
                {
                    "enrolled": enrolled,
                    "enrollment_id": enrollment.enrollment_id if enrollment else None,
                    "fingerprint": enrollment.fingerprint if enrollment else None,
                    "socket": str(sock),
                    "socket_present": sock.exists(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"未知 operatord 命令 {command!r}", file=sys.stderr)
    return 2
