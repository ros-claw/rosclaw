"""Tiny MCP stdio fixture used for process timeout and framing tests."""

from __future__ import annotations

import argparse
import json
import sys
import time


def _respond(request_id: int, result: dict) -> None:
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("healthy", "stall", "crash", "oversize", "malformed", "wrong-id"),
    )
    args = parser.parse_args()
    for raw in sys.stdin:
        message = json.loads(raw)
        request_id = message.get("id")
        method = message.get("method")
        if request_id is None:
            continue
        if method == "initialize":
            _respond(request_id, {"protocolVersion": "2024-11-05", "capabilities": {}})
        elif method == "tools/list":
            _respond(request_id, {"tools": [{"name": "capture_aligned_rgbd"}]})
        elif method == "tools/call":
            if args.mode == "stall":
                time.sleep(60)
            elif args.mode == "crash":
                return 23
            elif args.mode == "oversize":
                print("{" + "x" * (1024 * 1024 + 10), flush=True)
            elif args.mode == "malformed":
                print('{"jsonrpc": "2.0",', flush=True)
            elif args.mode == "wrong-id":
                print(
                    json.dumps({"jsonrpc": "2.0", "id": True, "result": {}}),
                    flush=True,
                )
            else:
                _respond(request_id, {"content": [{"type": "text", "text": "ok"}]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
