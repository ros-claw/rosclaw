"""Actual Linux local IPC contracts; no DDS, service, World or actuator."""

import base64
import importlib
import json
import os
import socket
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("probe_controller_ipc")


def packet():
    return {
        "schema_version": "rosclaw.probe_controller_ipc.v1",
        "kind": "backend_probe_lift_begin",
        "run_id": "synthetic_owned_run",
        "constraint_policy_hash": "b" * 64,
        "transaction_id": "a" * 64,
        "request_base64": base64.b64encode(b"not_an_executed_scene_request").decode(),
    }


def decode(module, raw):
    return module.decode_controller_packet(
        raw, run_id="synthetic_owned_run", constraint_policy_hash="b" * 64
    )


def test_real_seqpacket_peer_credentials_and_original_packet_roundtrip(module, tmp_path):
    tmp_path.chmod(0o700)
    server = module.ProbeControllerIPC(
        tmp_path / "instrument.sock", controller_pid=os.getpid(), controller_uid=os.getuid()
    )
    raw = json.dumps(packet()).encode()
    try:
        assert server.poll() is None
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.settimeout(0.2)
            client.connect(str(server.path))
            client.sendall(raw)
            received, peer = server.poll()
            assert received == raw and peer["pid"] == os.getpid() and peer["uid"] == os.getuid()
            assert peer["process_starttime"] == server.peer[2]
            kind, payload = decode(module, received)
            assert kind == "backend_probe_lift_begin"
            assert (
                base64.b64decode(payload["request"]["original_source_base64"])
                == b"not_an_executed_scene_request"
            )
            server.finish(b'{"authorization":false}')
            assert client.recv(128) == b'{"authorization":false}'
    finally:
        server.close()
    assert not server.path.exists()


def test_real_oversized_seqpacket_is_refused_without_truncating_original(module, tmp_path):
    tmp_path.chmod(0o700)
    server = module.ProbeControllerIPC(
        tmp_path / "instrument.sock", controller_pid=os.getpid(), controller_uid=os.getuid()
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.connect(str(server.path))
            client.sendall(b"x" * 16385)
            with pytest.raises(ValueError, match="oversized"):
                server.poll()
    finally:
        server.close()


@pytest.mark.parametrize(
    "fault",
    ["run", "policy", "duplicate", "actuator", "unknown", "bad_base64", "boolean_returncode"],
)
def test_wrong_binding_or_control_or_malformed_original_packet_is_refused(module, fault):
    p = packet()
    if fault == "run":
        p["run_id"] = "other_run"
    elif fault == "policy":
        p["constraint_policy_hash"] = "c" * 64
    elif fault == "actuator":
        p["kind"] = "robot_move"
    elif fault == "unknown":
        p["authorization"] = True
    elif fault == "bad_base64":
        p["request_base64"] = "!"
    elif fault == "boolean_returncode":
        p.update(
            kind="backend_probe_lift_ack",
            response_base64="ZGF0YTogdHJ1ZQ==",
            returncode=True,
            acknowledged_at_unix_ns=1,
        )
    raw = json.dumps(p).encode()
    if fault == "duplicate":
        raw = raw[:-1] + b',"run_id":"synthetic_owned_run"}'
    with pytest.raises(ValueError):
        decode(module, raw)


def test_existing_socket_path_or_nonprivate_directory_is_refused(module, tmp_path):
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match="private"):
        module.ProbeControllerIPC(
            tmp_path / "instrument.sock", controller_pid=os.getpid(), controller_uid=os.getuid()
        )
    tmp_path.chmod(0o700)
    path = tmp_path / "instrument.sock"
    path.write_text("original_other_source")
    with pytest.raises(ValueError, match="exclusive"):
        module.ProbeControllerIPC(path, controller_pid=os.getpid(), controller_uid=os.getuid())
    assert path.read_text() == "original_other_source"


def test_real_connected_client_cannot_hold_source_channel_past_deadline(module, tmp_path):
    tmp_path.chmod(0o700)
    server = module.ProbeControllerIPC(
        tmp_path / "instrument.sock", controller_pid=os.getpid(), controller_uid=os.getuid()
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.connect(str(server.path))
            assert server.poll() is None
            server.accepted_at -= 0.101
            with pytest.raises(ValueError, match="deadline expired"):
                server.poll()
    finally:
        server.close()


def test_actual_kernel_credentials_refuse_other_live_process_identity(module, tmp_path):
    tmp_path.chmod(0o700)
    other = subprocess.Popen(["sleep", "5"])
    try:
        server = module.ProbeControllerIPC(
            tmp_path / "instrument.sock", controller_pid=other.pid, controller_uid=os.getuid()
        )
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
                client.connect(str(server.path))
                client.sendall(json.dumps(packet()).encode())
                with pytest.raises(ValueError, match="credentials differ"):
                    server.poll()
        finally:
            server.close()
    finally:
        other.terminate()
        other.wait(timeout=2)


def test_array_packet_is_refused_with_bounded_source_error(module):
    with pytest.raises(ValueError, match="object required"):
        decode(module, b"[]")
