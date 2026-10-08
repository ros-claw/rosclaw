"""Private bounded original SIM instrument IPC, never a robot authorization.

Linux peer credentials pin one owned controller process. They do not prove
Gazebo ownership, publisher authentication or successful physical intervention.
This module performs no scene service, DDS operation or robot command.
"""

import base64
import hashlib
import os
import socket
import stat
import struct
import time
from pathlib import Path

from probe_scene_geometry import decode_scene_json


def process_identity(pid, uid):
    if type(pid) is not int or pid <= 0 or type(uid) is not int or uid < 0:
        raise ValueError("explicit live owned controller PID/UID required")
    source = Path(f"/proc/{pid}")
    if source.stat().st_uid != uid:
        raise ValueError("owned controller process UID differs")
    raw = (source / "stat").read_text()
    # comm may contain spaces and parentheses; starttime is field 22.
    fields = raw[raw.rfind(")") + 2 :].split()
    return pid, uid, int(fields[19])


def decode_controller_packet(raw, *, run_id, constraint_policy_hash):
    if type(raw) is not bytes or not 0 < len(raw) <= 16384:
        raise ValueError("bounded original instrument IPC packet required")
    packet = decode_scene_json(raw)
    if type(packet) is not dict:
        raise ValueError("original instrument IPC object required")
    keys = {
        "schema_version",
        "kind",
        "run_id",
        "constraint_policy_hash",
        "transaction_id",
        "request_base64",
    }
    if packet.get("kind") == "backend_probe_lift_ack":
        keys |= {"response_base64", "returncode", "acknowledged_at_unix_ns"}
        if "service_record_base64" in packet or "service_binary_sha256" in packet:
            keys |= {"service_record_base64", "service_binary_sha256"}
    elif packet.get("kind") != "backend_probe_lift_begin":
        raise ValueError("instrument IPC only accepts original lift begin/reply")
    if (
        set(packet) != keys
        or packet["schema_version"] != "rosclaw.probe_controller_ipc.v1"
        or packet["run_id"] != run_id
        or packet["constraint_policy_hash"] != constraint_policy_hash
        or type(packet["transaction_id"]) is not str
        or len(packet["transaction_id"]) != 64
        or any(c not in "0123456789abcdef" for c in packet["transaction_id"])
    ):
        raise ValueError("exact owned run/policy/transaction instrument IPC envelope required")

    def original(field, source_type):
        if type(packet[field]) is not str:
            raise ValueError("original instrument IPC base64 string required")
        try:
            source = base64.b64decode(packet[field], validate=True)
        except ValueError as exc:
            raise ValueError("invalid original instrument IPC base64") from exc
        if not 0 < len(source) <= 4096:
            raise ValueError("bounded original instrument service bytes required")
        return {
            "original_source_base64": packet[field],
            "original_source_sha256": hashlib.sha256(source).hexdigest(),
            "original_size_bytes": len(source),
            "source_bytes_complete": True,
            "source_type": source_type,
        }

    payload = {
        "transaction_id": packet["transaction_id"],
        "request": original("request_base64", "gz.msgs.Pose_protobuf_text"),
    }
    if packet["kind"] == "backend_probe_lift_ack":
        if (
            type(packet["returncode"]) is not int
            or type(packet["acknowledged_at_unix_ns"]) is not int
            or packet["acknowledged_at_unix_ns"] <= 0
        ):
            raise ValueError("original integer service exit/completion clock required")
        payload.update(
            response=original("response_base64", "gz.msgs.Boolean_protobuf_text"),
            returncode=packet["returncode"],
            acknowledged_at_unix_ns=packet["acknowledged_at_unix_ns"],
        )
        if "service_record_base64" in packet:
            if (
                type(packet["service_binary_sha256"]) is not str
                or len(packet["service_binary_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in packet["service_binary_sha256"])
            ):
                raise ValueError("frozen original instrument service binary SHA required")
            payload.update(
                original_service_record=original(
                    "service_record_base64", "owned_gazebo_instrument_RPC_original_json"
                ),
                service_binary_sha256=packet["service_binary_sha256"],
            )
    return packet["kind"], payload


class ProbeControllerIPC:
    """One nonblocking private SOCK_SEQPACKET listener, max one packet per poll."""

    def __init__(self, path, *, controller_pid, controller_uid):
        self.path = Path(path)
        parent = self.path.parent
        info = parent.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or parent.is_symlink()
            or parent.resolve() != parent.absolute()
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
            or len(os.fsencode(self.path)) > 100
            or os.path.lexists(self.path)
        ):
            raise ValueError("exclusive short private owned instrument IPC directory required")
        self.peer = process_identity(controller_pid, controller_uid)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.connection = None
        self.accepted_at = None
        try:
            self.socket.bind(str(self.path))
            os.chmod(self.path, 0o600)
            self.identity = (self.path.stat().st_dev, self.path.stat().st_ino)
            self.socket.listen(1)
            self.socket.setblocking(False)
        except BaseException:
            self.socket.close()
            raise

    def poll(self):
        if self.connection is None:
            try:
                self.connection, _ = self.socket.accept()
            except BlockingIOError:
                return None
            self.connection.setblocking(False)
            self.accepted_at = time.monotonic()
        if time.monotonic() - self.accepted_at >= 0.1:
            self.finish(b'{"accepted":false,"authorization":false}')
            raise ValueError("instrument IPC connected packet deadline expired")
        pid, uid, _ = struct.unpack(
            "3i", self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if (pid, uid) != self.peer[:2] or process_identity(pid, uid) != self.peer:
            self.finish(b'{"accepted":false,"authorization":false}')
            raise ValueError("instrument IPC live process credentials differ")
        try:
            raw, _, flags, _ = self.connection.recvmsg(16384)
        except BlockingIOError:
            return None
        if flags & socket.MSG_TRUNC or not raw:
            self.finish(b'{"accepted":false,"authorization":false}')
            raise ValueError("instrument IPC empty/disconnected/oversized original packet")
        return raw, {"pid": pid, "uid": uid, "process_starttime": self.peer[2]}

    def finish(self, response):
        if self.connection is not None:
            try:
                if self.connection.send(response) != len(response):
                    raise ValueError("instrument IPC response incomplete")
            finally:
                self.connection.close()
                self.connection = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.socket.close()
        if (
            self.path.exists()
            and (self.path.stat().st_dev, self.path.stat().st_ino) == self.identity
        ):
            self.path.unlink()
