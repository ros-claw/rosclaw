#!/usr/bin/env python3
"""Offline CLI example for the canonical pure perception quality modules.

Usage:
    python examples/perception_sensor_quality.py --case scan --input ABS_JSON \
        --reference-time-ns SIGNED_INT
    python examples/perception_sensor_quality.py --case cloud --input ABS_JSON \
        [--reference-time-ns SIGNED_INT]

The JSON input is an ordinary nested dict describing a LaserScan-like or
PointCloud2-like message. Nested dicts are converted into attribute objects
(duck typing), and a cloud ``data`` integer list becomes a bytes-compatible
object. Exactly one canonical analyzer call is made and one JSON result is
printed to stdout. No ROS, no middleware, no network, no actuation.
"""

import argparse
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rosclaw.perception import cloud_quality, scan_quality  # noqa: E402


def _to_message(node):
    """Convert nested JSON dicts into duck-typed attribute objects."""
    if isinstance(node, dict):
        msg = types.SimpleNamespace()
        for key, value in node.items():
            setattr(msg, key, _to_message(value))
        return msg
    if isinstance(node, list):
        return [_to_message(item) for item in node]
    return node


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline perception sensor quality example")
    parser.add_argument("--case", choices=["scan", "cloud"], required=True)
    parser.add_argument("--input", required=True, help="Absolute path to the JSON input file")
    parser.add_argument(
        "--reference-time-ns",
        type=int,
        default=None,
        help="Explicit signed reference clock in ns (required for scan, optional for cloud)",
    )
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text())

    if args.case == "scan":
        if args.reference_time_ns is None:
            parser.error("--reference-time-ns is required for --case scan")
        message = _to_message(payload)
        result = scan_quality.analyze_scan(message, args.reference_time_ns)
    else:
        message = _to_message(payload)
        if isinstance(getattr(message, "data", None), list):
            message.data = bytes(message.data)
        result = cloud_quality.analyze_cloud(message)

    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
