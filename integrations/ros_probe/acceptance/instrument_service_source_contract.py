"""Installed SDK original protobuf parsing only; no transport Node or service."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from instrument_service_evidence import parse_original_service_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=False)
    original = args.directory / "original-source"
    original.mkdir()
    hashes = {}
    sources = [
        Path(__file__),
        Path(__file__).with_name("instrument_service_evidence.py"),
        *Path(__file__).with_name("backend_instrument_service").glob("*"),
    ]
    for source in sources:
        raw = source.read_bytes()
        (original / source.name).write_bytes(raw)
        hashes[str(source)] = hashlib.sha256(raw).hexdigest()
    binary = args.binary.read_bytes()
    binary_sha = hashlib.sha256(binary).hexdigest()
    (original / "executed-owned-instrument-service").write_bytes(binary)
    request = (
        b'name: "owned_instrument" position { x: 6 y: 0 z: 10 } orientation { w: 1 x: 0 y: 0 z: 0 }'
    )
    base = [
        str(args.binary),
        "ros_expert",
        "owned_instrument",
        "active_robot",
        "6",
        "0",
        "10",
        "rosclaw_backend_" + "a" * 32,
        "--source-validate-only",
    ]
    cases = [
        ("actual_sdk_request_and_true_boolean_wire", request, "1001", base, True),
        ("actual_sdk_false_boolean_wire", request, "1000", base, True),
        ("request_only_no_service_reply", request, "-", base, True),
        ("malformed_reply_wire", request, "ff", base, False),
        (
            "active_robot_target_refused",
            request.replace(b"owned_instrument", b"active_robot"),
            "-",
            base,
            False,
        ),
        (
            "other_instrument_target_refused",
            request.replace(b"owned_instrument", b"other_instrument"),
            "-",
            base,
            False,
        ),
        ("wrong_horizontal_target", request.replace(b"x: 6", b"x: 7"), "-", base, False),
        ("wrong_lift_target", request.replace(b"z: 10", b"z: 11"), "-", base, False),
        ("header_injection", request + b" header { stamp { sec: 1 } }", "-", base, False),
        ("role_alias", request, "-", [*base[:3], "owned_instrument", *base[4:]], False),
        ("nan_declaration", request, "-", [*base[:4], "nan", *base[5:]], False),
        ("unsafe_world_name", request, "-", [base[0], "world;other", *base[2:]], False),
        ("oversized_original_request", b"x" * 8193, "-", base, False),
    ]
    reports = []
    for name, raw, reply, argv, success in cases:
        directory = args.directory / name
        directory.mkdir()
        source = raw.hex().encode() + b" " + reply.encode() + b"\n"
        (directory / "original-input.hex-line").write_bytes(source)
        checked = subprocess.run(argv, input=source, capture_output=True, timeout=5)
        (directory / "stdout.jsonl").write_bytes(checked.stdout)
        (directory / "stderr.txt").write_bytes(checked.stderr)
        (directory / "invocation.json").write_text(json.dumps(argv) + "\n")
        if (checked.returncode == 0) != success:
            raise ValueError("actual SDK source-only parser result differs: " + name)
        records = [json.loads(line) for line in checked.stdout.splitlines()]
        if any(
            r.get("transport_node_started") is True
            or r.get("service_executed") is True
            or r.get("authorization") is not False
            for r in records
        ):
            raise ValueError(
                "source-only parser started transport/service or claimed authorization"
            )
        if success:
            if len(records) != 2 or bytes.fromhex(records[1]["request_text_hex"]) != raw:
                raise ValueError("actual SDK parser original source request correspondence differs")
            if (
                name == "actual_sdk_request_and_true_boolean_wire"
                and records[1]["response_data"] is not True
            ):
                raise ValueError("actual installed SDK Boolean true wire not decoded")
            if name == "actual_sdk_false_boolean_wire" and records[1]["response_data"] is not False:
                raise ValueError("actual installed SDK Boolean false wire not decoded")
            if reply != "-" and records[1]["response_protobuf_hex"] != reply:
                raise ValueError("actual SDK Boolean source wire changed")
            try:
                parse_original_service_record(checked.stdout.splitlines()[1], None)
            except ValueError as exc:
                if "successful actual instrument RPC" not in str(exc):
                    raise
            else:
                raise ValueError("source-only SDK Boolean was misclassified as an actual service")
        reports.append(
            {
                "case": name,
                "status": "PASS_EXPECTED_SOURCE_RESULT",
                "returncode": checked.returncode,
                "source_only": True,
            }
        )
    if args.binary.read_bytes() != binary or any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in hashes.items()
    ):
        raise ValueError("original installed source parser bytes changed during execution")
    review = {
        "status": "PASS_INSTALLED_SDK_SOURCE_ONLY_INSTRUMENT_PARSER",
        "cases": reports,
        "original_source_hashes": hashes,
        "executed_binary_sha256": binary_sha,
        "transport_Node_started": False,
        "World_started": False,
        "scene_service_executed": False,
        "actuator_or_Native_task_started": False,
        "physical_acceptance": "NOT_RUN",
        "authorization": False,
    }
    (args.directory / "source-contract-review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(
        json.dumps(
            {"status": review["status"], "cases": len(reports), "physical_acceptance": "NOT_RUN"}
        )
    )


if __name__ == "__main__":
    main()
