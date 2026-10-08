"""Retained original owned instrument RPC bytes and installed SDK replay.

An RPC outcome only arms the measured probe state machine. This module starts
no World, transport Node or service, and never grants robot authorization.
"""

import hashlib
import re
import subprocess
from pathlib import Path

from backend_probe_world import bounded_source
from probe_lift_evidence import derive_lift_ack, lift_request
from probe_scene_geometry import decode_scene_json


def parse_original_service_record(raw, policy):
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise ValueError("bounded original instrument RPC source record required")
    record = decode_scene_json(raw)
    keys = {
        "request_text_hex",
        "request_protobuf_hex",
        "response_protobuf_hex",
        "response_text_hex",
        "transport_returned",
        "service_result",
        "response_data",
        "acknowledged_at_unix_ns",
        "service_executed",
        "authorization",
    }
    if type(record) is not dict or set(record) != keys:
        raise ValueError("exact original owned instrument RPC source fields required")
    for key in ("transport_returned", "service_result", "response_data", "service_executed"):
        if record[key] is not True:
            raise ValueError(
                "original successful actual instrument RPC required; source-only reply cannot arm lift"
            )
    if record["authorization"] is not False:
        raise ValueError("instrument RPC source cannot grant robot authorization")
    sources = {}
    for key in (
        "request_text_hex",
        "request_protobuf_hex",
        "response_protobuf_hex",
        "response_text_hex",
    ):
        value = record[key]
        if (
            type(value) is not str
            or not 0 < len(value) <= 8192
            or len(value) % 2
            or not re.fullmatch(r"[a-f0-9]+", value)
        ):
            raise ValueError("bounded original instrument protobuf source hex required")
        sources[key] = bytes.fromhex(value)
    if sources["request_text_hex"] != lift_request(policy):
        raise ValueError("original RPC request differs from frozen instrument-only target")
    # Installed gz-msgs10 source contracts verify this exact header-free true
    # Boolean wire. Extra/unknown/header fields are not silently normalized.
    if sources["response_protobuf_hex"] != b"\x10\x01":
        raise ValueError("exact qualified SDK Boolean true wire required")
    ack = derive_lift_ack(
        policy,
        request_bytes=sources["request_text_hex"],
        response_bytes=sources["response_text_hex"],
        returncode=0,
        acknowledged_at_unix_ns=record["acknowledged_at_unix_ns"],
    )
    return record, sources, ack


def verify_service_sdk_projection(
    raw, policy, *, binary_path, binary_sha256, robot_model_name, partition
):
    """Reparse retained wire bytes with frozen installed SDK, source-only mode."""
    record, sources, ack = parse_original_service_record(raw, policy)
    path = Path(binary_path)
    binary = bounded_source(path, 100_000_000)
    if not binary.startswith(b"\x7fELF") or hashlib.sha256(binary).hexdigest() != binary_sha256:
        raise ValueError("frozen actual instrument SDK parser executable SHA differs")
    base = policy["native_policy"]
    argv = [
        str(path),
        base["world_name"],
        base["contact_policy"]["model_name"],
        robot_model_name,
        *[f"{v:.17g}" for v in (*policy["probe_xy"], policy["lift_z_m"])],
        partition,
        "--source-validate-only",
    ]
    input_bytes = (
        sources["request_text_hex"].hex().encode()
        + b" "
        + sources["response_protobuf_hex"].hex().encode()
        + b"\n"
    )
    result = subprocess.run(argv, input=input_bytes, capture_output=True, timeout=5)
    if result.returncode or not 0 < len(result.stdout) <= 65536 or len(result.stderr) > 65536:
        raise ValueError("installed source-only instrument SDK replay failed")
    outputs = [decode_scene_json(line) for line in result.stdout.splitlines()]
    if (
        len(outputs) != 2
        or outputs[0].get("transport_node_started") is not False
        or outputs[1].get("service_executed") is not False
        or any(output.get("authorization") is not False for output in outputs)
        or any(
            outputs[1].get(k) != record[k]
            for k in (
                "request_text_hex",
                "request_protobuf_hex",
                "response_protobuf_hex",
                "response_text_hex",
                "response_data",
            )
        )
        or bounded_source(path, 100_000_000) != binary
    ):
        raise ValueError("original instrument RPC wire and SDK-derived projections differ")
    return {
        "evidence_role": "installed_SDK_original_RPC_wire_correspondence_not_measured_probe_or_permission",
        "original_service_record_sha256": hashlib.sha256(raw).hexdigest(),
        "original_request_wire_sha256": hashlib.sha256(sources["request_protobuf_hex"]).hexdigest(),
        "original_reply_wire_sha256": hashlib.sha256(sources["response_protobuf_hex"]).hexdigest(),
        "source_parser_binary_sha256": binary_sha256,
        "lift_ack": ack,
        "original_source_sdk_projection_verified": True,
        "SDK_replay_transport_Node_or_service_started": False,
        "backend_health_admitted": False,
        "physical_acceptance": "NOT_VERIFIED",
        "authorization": False,
    }
