"""Protocol-level tests for the rosclawd local control socket."""

from __future__ import annotations

import socket
import struct
from typing import Any

import pytest

from rosclaw.daemon.protocol import (
    DAEMON_PROTOCOL_VERSION,
    MAX_FRAME_BYTES,
    DaemonProtocolError,
    decode_request,
    encode_frame,
    make_request,
    receive_frame,
)


def test_request_round_trip_is_versioned_and_bounded() -> None:
    request = make_request(
        "runtime.status",
        {},
        request_id="request-1",
    )

    parsed = decode_request(request)

    assert parsed.protocol_version == DAEMON_PROTOCOL_VERSION
    assert parsed.request_id == "request-1"
    assert parsed.method == "runtime.status"
    assert parsed.params == {}


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({}, "INVALID_PROTOCOL"),
        (
            {
                "protocol_version": "rosclaw.daemon.v0",
                "request_id": "request-1",
                "method": "runtime.status",
                "params": {},
            },
            "UNSUPPORTED_PROTOCOL",
        ),
        (
            {
                "protocol_version": DAEMON_PROTOCOL_VERSION,
                "request_id": "",
                "method": "runtime.status",
                "params": {},
            },
            "INVALID_REQUEST_ID",
        ),
        (
            {
                "protocol_version": DAEMON_PROTOCOL_VERSION,
                "request_id": "request-1",
                "method": "runtime.status",
                "params": [],
            },
            "INVALID_PARAMS",
        ),
    ],
)
def test_invalid_request_shapes_fail_closed(payload: dict[str, Any], code: str) -> None:
    with pytest.raises(DaemonProtocolError) as error:
        decode_request(payload)

    assert error.value.code == code


def test_receive_frame_rejects_oversized_payload_before_reading_body() -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(struct.pack("!I", MAX_FRAME_BYTES + 1))
        with pytest.raises(DaemonProtocolError) as error:
            receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()

    assert error.value.code == "FRAME_TOO_LARGE"


def test_receive_frame_rejects_non_object_json() -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(encode_frame(["not", "an", "object"]))
        with pytest.raises(DaemonProtocolError) as error:
            receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()

    assert error.value.code == "INVALID_MESSAGE"
