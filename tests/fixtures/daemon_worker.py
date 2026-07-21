"""Subprocess fixture speaking the rosclawd Adapter worker health protocol."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

PROTOCOL = "rosclaw.adapter.worker.v1"


def _emit(message_type: str) -> None:
    print(
        json.dumps({"protocol_version": PROTOCOL, "type": message_type}),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("healthy", "crash", "stall", "invalid", "oversize"),
        required=True,
    )
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("--after", type=float, default=0.1)
    parser.add_argument("--ignore-term", action="store_true")
    args = parser.parse_args()
    if args.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if args.mode == "invalid":
        print("not-json", flush=True)
    elif args.mode == "oversize":
        print("x" * (64 * 1024 + 1), flush=True)
    else:
        _emit("ready")
    started = time.monotonic()
    while True:
        if args.mode == "crash" and time.monotonic() - started >= args.after:
            return 17
        if args.mode not in {"stall", "invalid", "oversize"}:
            _emit("heartbeat")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
