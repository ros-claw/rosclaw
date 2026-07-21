"""Command-line entry points for rosclawd and its northbound client."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import stat
import sys
from pathlib import Path
from typing import Any

from rosclaw.daemon.client import (
    DaemonClient,
    DaemonClientError,
    get_daemon_socket_path,
)


def build_daemon_runtime(robot_id: str) -> Any:
    """Build the control-plane Runtime without asynchronous knowledge services."""

    from rosclaw.core.runtime import Runtime, RuntimeConfig

    runtime = Runtime(
        RuntimeConfig(
            robot_id=robot_id,
            enable_memory=False,
            enable_practice=False,
            enable_swarm=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
        )
    )
    runtime.initialize()
    from rosclaw.robot_pack.runtime_loader import load_daemon_robot_pack

    load_daemon_robot_pack(runtime, robot_id=robot_id)
    runtime.start()
    return runtime


def run_daemon(
    *,
    socket_path: str | Path | None,
    socket_mode: int,
    socket_group: str | None,
    robot_id: str,
    log_level: str,
    ledger_path: str | Path | None = None,
    ledger_key_path: str | Path | None = None,
) -> int:
    """Run rosclawd in the foreground for systemd or an operator terminal."""

    from rosclaw.daemon.ledger import (
        DaemonLedger,
        get_daemon_ledger_key_path,
        get_daemon_ledger_path,
    )
    from rosclaw.daemon.server import RosclawDaemon
    from rosclaw.daemon.service import DaemonControlPlane

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    with DaemonLedger(
        get_daemon_ledger_path(ledger_path),
        key_path=get_daemon_ledger_key_path(ledger_key_path),
    ) as ledger:
        runtime = build_daemon_runtime(robot_id)
        service: DaemonControlPlane | None = None
        try:
            service = DaemonControlPlane(runtime=runtime, ledger=ledger)
            daemon = RosclawDaemon(
                service=service,
                socket_path=socket_path,
                socket_mode=socket_mode,
                socket_group=socket_group,
            )

            previous_handlers: dict[signal.Signals, Any] = {}

            def _request_shutdown(_signum: int, _frame: Any) -> None:
                daemon.request_shutdown()

            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _request_shutdown)
            try:
                daemon.serve_forever()
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
        finally:
            if service is not None:
                service.close()
            else:
                with contextlib.suppress(Exception):
                    runtime.request_emergency_stop(
                        "rosclawd startup failed after Runtime initialization",
                        source="rosclawd.startup_failure",
                        timeout_sec=1.0,
                    )
                with contextlib.suppress(Exception):
                    runtime.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dedicated ``rosclawd`` foreground entry point."""

    parser = argparse.ArgumentParser(
        prog="rosclawd",
        description="ROSClaw least-privilege physical execution daemon",
    )
    _add_serve_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run_daemon(
            socket_path=args.socket,
            socket_mode=args.socket_mode,
            socket_group=args.socket_group,
            robot_id=args.robot_id,
            log_level=args.log_level,
            ledger_path=args.ledger,
            ledger_key_path=args.ledger_key,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[rosclawd] startup failed: {exc}", file=sys.stderr)
        return 1


def dispatch_daemon_argv(argv: list[str]) -> int | None:
    """Handle ``rosclaw daemon`` without importing the large legacy CLI."""

    if not argv or argv[0] != "daemon":
        return None
    parser = _build_client_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "daemon_handler", None)
    if not callable(handler):
        parser.print_help()
        return 1
    try:
        return int(handler(args))
    except DaemonClientError as exc:
        payload = {
            "ok": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        }
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"[ROSClaw] {exc.code}: {exc.message}", file=sys.stderr)
        return 2


def _build_client_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw")
    daemon_parser = parser.add_subparsers(dest="command").add_parser(
        "daemon",
        help="Inspect or call the local rosclawd control plane",
    )
    subparsers = daemon_parser.add_subparsers(dest="daemon_command")

    serve = subparsers.add_parser("serve", help="Run rosclawd in the foreground")
    _add_serve_arguments(serve)
    serve.set_defaults(daemon_handler=_cmd_serve)

    status = subparsers.add_parser("status", help="Show daemon and privilege-boundary status")
    _add_client_arguments(status)
    status.set_defaults(daemon_handler=_cmd_status)

    acknowledge = subparsers.add_parser(
        "acknowledge-recovery",
        help="Persist daemon-UID review of interrupted REAL action evidence",
    )
    acknowledge.add_argument("--reason", required=True)
    _add_client_arguments(acknowledge)
    acknowledge.set_defaults(daemon_handler=_cmd_acknowledge_recovery)

    request = subparsers.add_parser(
        "request-action",
        help="Submit a canonical ActionEnvelope JSON file through rosclawd",
    )
    request.add_argument("action_file", help="ActionEnvelope JSON path, or - for stdin")
    request.add_argument(
        "--wait",
        type=float,
        default=5.0,
        help="Seconds to wait for a terminal receipt; 0 returns the queue ticket",
    )
    _add_client_arguments(request)
    request.set_defaults(daemon_handler=_cmd_request_action)

    action_status = subparsers.add_parser("action-status", help="Read queued action status")
    action_status.add_argument("action_id")
    _add_client_arguments(action_status)
    action_status.set_defaults(daemon_handler=_cmd_action_status)

    receipt = subparsers.add_parser("receipt", help="Read a daemon ExecutionReceipt")
    receipt.add_argument("action_id")
    _add_client_arguments(receipt)
    receipt.set_defaults(daemon_handler=_cmd_receipt)

    cancel = subparsers.add_parser("cancel", help="Cancel an action before dispatch")
    cancel.add_argument("action_id")
    _add_client_arguments(cancel)
    cancel.set_defaults(daemon_handler=_cmd_cancel)

    emergency = subparsers.add_parser(
        "emergency-stop",
        help="Request acknowledged emergency stop through rosclawd",
    )
    emergency.add_argument("--reason", required=True)
    emergency.add_argument("--stop-timeout", type=float, default=1.0)
    _add_client_arguments(emergency)
    emergency.set_defaults(daemon_handler=_cmd_emergency_stop)

    security = subparsers.add_parser(
        "security-check",
        help="Check whether this client process lacks direct southbound access",
    )
    _add_client_arguments(security)
    security.set_defaults(daemon_handler=_cmd_security_check)

    stop = subparsers.add_parser(
        "stop",
        help="Request shutdown (allowed only to the rosclawd service UID)",
    )
    _add_client_arguments(stop)
    stop.set_defaults(daemon_handler=_cmd_stop)
    return parser


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--socket",
        default=os.environ.get("ROSCLAW_DAEMON_SOCKET"),
        help="Unix socket path (default: $ROSCLAW_HOME/run/rosclawd.sock)",
    )
    parser.add_argument(
        "--socket-mode",
        type=_parse_octal_mode,
        default=0o600,
        help="Socket mode in octal (default: 0600; use 0660 with a client group)",
    )
    parser.add_argument(
        "--socket-group",
        default=os.environ.get("ROSCLAW_DAEMON_SOCKET_GROUP"),
        help="Optional group owning the client socket",
    )
    parser.add_argument(
        "--robot-id",
        default=os.environ.get("ROSCLAW_ROBOT_ID", "rosclaw_default"),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("ROSCLAW_LOG_LEVEL", "INFO").upper(),
    )
    parser.add_argument(
        "--ledger",
        default=os.environ.get("ROSCLAW_DAEMON_LEDGER"),
        help="Durable daemon ledger path (default: $ROSCLAW_HOME/state/daemon/ledger.sqlite3)",
    )
    parser.add_argument(
        "--ledger-key",
        default=os.environ.get("ROSCLAW_DAEMON_LEDGER_KEY"),
        help="Daemon-private ledger HMAC key path",
    )


def _add_client_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--socket",
        default=os.environ.get("ROSCLAW_DAEMON_SOCKET"),
        help="Unix socket path",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")


def _client(args: argparse.Namespace) -> DaemonClient:
    return DaemonClient(socket_path=args.socket, timeout_sec=args.timeout)


def _cmd_serve(args: argparse.Namespace) -> int:
    return run_daemon(
        socket_path=args.socket,
        socket_mode=args.socket_mode,
        socket_group=args.socket_group,
        robot_id=args.robot_id,
        log_level=args.log_level,
        ledger_path=args.ledger,
        ledger_key_path=args.ledger_key,
    )


def _cmd_status(args: argparse.Namespace) -> int:
    payload = _client(args).get_runtime_status()
    return _print_payload(args, payload)


def _cmd_acknowledge_recovery(args: argparse.Namespace) -> int:
    payload = _client(args).acknowledge_recovery(args.reason)
    _print_payload(args, payload)
    return 0 if payload.get("recovery_required") is False else 3


def _cmd_request_action(args: argparse.Namespace) -> int:
    if args.action_file == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(args.action_file).read_text(encoding="utf-8")
    try:
        action = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DaemonClientError("INVALID_ACTION_JSON", str(exc)) from exc
    if not isinstance(action, dict):
        raise DaemonClientError("INVALID_ACTION_JSON", "ActionEnvelope must be a JSON object")
    client = _client(args)
    payload = client.request_action(action)
    if args.wait > 0:
        payload = client.wait_for_action(
            str(payload["action_id"]),
            timeout_sec=args.wait,
        )
    result = _print_payload(args, payload)
    receipt = payload.get("receipt")
    if isinstance(receipt, dict) and receipt.get("final_state") not in {
        "COMPLETED",
        "DEGRADED",
    }:
        return 3
    return result


def _cmd_action_status(args: argparse.Namespace) -> int:
    return _print_payload(args, _client(args).get_action_status(args.action_id))


def _cmd_receipt(args: argparse.Namespace) -> int:
    return _print_payload(args, _client(args).get_execution_receipt(args.action_id))


def _cmd_cancel(args: argparse.Namespace) -> int:
    payload = _client(args).cancel_action(args.action_id)
    _print_payload(args, payload)
    return 0 if payload.get("cancelled") is True else 3


def _cmd_emergency_stop(args: argparse.Namespace) -> int:
    payload = _client(args).emergency_stop(
        args.reason,
        source="rosclaw.daemon.cli",
        timeout_sec=args.stop_timeout,
    )
    _print_payload(args, payload)
    physically_confirmed = (
        payload.get("stopped") is True and payload.get("physical_stop_observed") is True
    )
    return 0 if physically_confirmed else 3


def _cmd_security_check(args: argparse.Namespace) -> int:
    client = _client(args)
    status = client.get_runtime_status()
    socket_metadata = get_daemon_socket_path(args.socket).stat()
    writable_devices = [
        str(path)
        for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/serial/by-id/*")
        for path in Path("/").glob(pattern.lstrip("/"))
        if os.access(path, os.R_OK | os.W_OK)
    ]
    group_ids = [os.getegid(), *os.getgroups()]
    group_names = _group_names(group_ids)
    check = {
        "schema_version": "rosclaw.daemon.security_check.v1",
        "client_pid": os.getpid(),
        "client_uid": os.geteuid(),
        "client_groups": group_names,
        "daemon_pid": status.get("daemon_pid"),
        "daemon_uid": status.get("daemon_uid"),
        "process_separated": status.get("daemon_pid") != os.getpid(),
        "privilege_separated": status.get("daemon_uid") != os.geteuid(),
        "socket_path": str(get_daemon_socket_path(args.socket)),
        "socket_mode": f"{stat.S_IMODE(socket_metadata.st_mode):04o}",
        "socket_world_writable": bool(stat.S_IMODE(socket_metadata.st_mode) & stat.S_IWOTH),
        "client_in_dialout": "dialout" in group_names,
        "writable_serial_devices": writable_devices,
    }
    check["boundary_ready"] = bool(
        check["process_separated"]
        and check["privilege_separated"]
        and not check["socket_world_writable"]
        and not check["client_in_dialout"]
        and not check["writable_serial_devices"]
    )
    _print_payload(args, check)
    return 0 if check["boundary_ready"] else 3


def _cmd_stop(args: argparse.Namespace) -> int:
    return _print_payload(args, _client(args).shutdown())


def _print_payload(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _parse_octal_mode(value: str) -> int:
    try:
        mode = int(value, 8)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("socket mode must be octal, e.g. 0600 or 0660") from exc
    if mode < 0 or mode > 0o777 or mode & 0o007 or mode & 0o600 != 0o600:
        raise argparse.ArgumentTypeError(
            "socket mode must grant owner read/write and must not be world-accessible"
        )
    return mode


def _group_names(group_ids: list[int]) -> list[str]:
    import grp

    names: list[str] = []
    for group_id in group_ids:
        try:
            names.append(grp.getgrgid(group_id).gr_name)
        except KeyError:
            names.append(str(group_id))
    return sorted(set(names))


__all__ = [
    "build_daemon_runtime",
    "dispatch_daemon_argv",
    "main",
    "run_daemon",
]


if __name__ == "__main__":
    raise SystemExit(main())
