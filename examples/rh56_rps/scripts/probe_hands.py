#!/usr/bin/env python3
"""Probe RH56 hand wiring without moving the hands.

Reads the HAND_ID holding register on each /dev/ttyUSB* port for slave ids
1 (left) and 2 (right).  Use this to verify which hand is on which port
before running hand-test / full modes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the demo package importable when run standalone.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rosclaw_rps.hand.port_scanner import discover_hands


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe RH56 hand serial wiring")
    parser.add_argument(
        "--glob",
        default="/dev/ttyUSB*",
        help="Serial device glob pattern",
    )
    parser.add_argument(
        "--left-id",
        type=int,
        default=1,
        help="Expected Modbus slave id for the left hand",
    )
    parser.add_argument(
        "--right-id",
        type=int,
        default=2,
        help="Expected Modbus slave id for the right hand",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show debug logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        mapping = discover_hands(
            port_glob=args.glob,
            slave_ids=(args.left_id, args.right_id),
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print("\nDiscovered hands:")
    left_port = mapping.get(args.left_id)
    right_port = mapping.get(args.right_id)
    print(f"  left  (id={args.left_id}):  {left_port or 'NOT FOUND'}")
    print(f"  right (id={args.right_id}): {right_port or 'NOT FOUND'}")

    if not left_port or not right_port:
        print("\nBoth hands were not detected. Check USB cables/adapters and power.")
        return 1

    print("\nTo use dynamic port assignment, set in your config:")
    print(f"  hand.port: auto")
    print(f"  referee.port: auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
