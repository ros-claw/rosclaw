"""Trusted local Operator Broker command surface.

This command must run as the rosclawd service UID (or an equivalent trusted
operator process). It never prints proposal challenges or execution permits.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from rosclaw.daemon.client import DaemonClient, DaemonClientError


def dispatch_operator_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "operator":
        return None
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "operator_handler", None)
    if not callable(handler):
        parser.print_help()
        return 1
    try:
        return int(handler(args))
    except DaemonClientError as exc:
        payload = {
            "ok": False,
            "error": {"code": exc.code, "message": exc.message, "details": exc.details},
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"[ROSClaw Operator] {exc.code}: {exc.message}")
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw")
    operator = parser.add_subparsers(dest="command").add_parser(
        "operator", help="Review daemon-owned physical action proposals"
    )
    commands = operator.add_subparsers(dest="operator_command")

    pending = commands.add_parser("pending", help="List pending exact-action proposals")
    _add_client_arguments(pending)
    pending.set_defaults(operator_handler=_cmd_pending)

    status = commands.add_parser("status", help="Read one proposal lifecycle")
    status.add_argument("request_id")
    _add_client_arguments(status)
    status.set_defaults(operator_handler=_cmd_status)

    decide = commands.add_parser("decide", help="Accept or decline one displayed proposal")
    decide.add_argument("request_id")
    choice = decide.add_mutually_exclusive_group(required=True)
    choice.add_argument("--accept", action="store_true")
    choice.add_argument("--decline", action="store_true")
    decide.add_argument("--principal-id", required=True)
    decide.add_argument("--reason", required=True)
    decide.add_argument("--channel", default="operator_cli")
    decide.add_argument(
        "--wait",
        type=float,
        default=300.0,
        help="Seconds to supervise and renew the accepted action lease; 0 returns immediately",
    )
    _add_client_arguments(decide)
    decide.set_defaults(operator_handler=_cmd_decide)
    return parser


def _add_client_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--socket")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")


def _client(args: argparse.Namespace) -> DaemonClient:
    return DaemonClient(socket_path=args.socket, timeout_sec=args.timeout)


def _cmd_pending(args: argparse.Namespace) -> int:
    payload = _client(args).list_pending_operator_proposals()
    return _print(args, _redact_challenges(payload))


def _cmd_status(args: argparse.Namespace) -> int:
    return _print(args, _client(args).get_operator_proposal(args.request_id))


def _cmd_decide(args: argparse.Namespace) -> int:
    """已移除（二次复核 R1）：决定只属于 rosclaw-operatord。

    旧行为（daemon 管理终端直决）正是二次复核要消除的旁路——唯一
    合法凭证是 operatord 用 Ed25519 私钥签的 OperatorDecisionProofV1
    （内含真实前台 Y/N 与 daemon 一次性 challenge）。
    """
    raise DaemonClientError(
        "DECISION_CHANNEL_MOVED",
        "proposal decisions belong to rosclaw-operatord (R1): "
        "`rosclaw operatord enroll && rosclaw operatord register-daemon && "
        "rosclaw operatord start`, then decide from the TUI/operatord channel. "
        "This admin CLI is read-only (pending/status).",
    )


def _redact_challenges(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _redact_challenges(item)
            for key, item in value.items()
            if key != "challenge_nonce" and "permit" not in key.lower()
        }
    if isinstance(value, list):
        return [_redact_challenges(item) for item in value]
    return value


def _print(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


__all__ = ["dispatch_operator_argv"]
