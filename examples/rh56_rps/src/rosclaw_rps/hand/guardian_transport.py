"""Client transport that talks to the RH56 serial guardian daemon.

The guardian keeps real ``/dev/ttyUSB*`` devices open in a long-lived process.
The demo connects to a Unix domain socket per hand and sends Modbus requests.
This avoids the CH340 close/reopen bug because the physical serial port is never
closed once the guardian has opened it.
"""
from __future__ import annotations

import socket
import struct
from typing import Optional


class GuardianTransport:
    """Transport that forwards Modbus traffic to a serial guardian via Unix socket.

    The daemon owns the real ``serial.Serial`` object and returns a
    length-prefixed response for every request.  This transport performs a
    full request-response transaction on every ``write`` and returns the
    cached response on the next ``read``.  This matches the controller's
    pattern of ``write(...)`` followed by ``read(...)`` and also consumes
    write-only configuration responses (e.g. speed/force setup) that the
    controller never explicitly reads.
    """

    def __init__(self, socket_path: str, timeout_s: float = 0.3):
        self._socket_path = socket_path
        self._timeout_s = timeout_s
        self._last_response: Optional[bytes] = None

    def open(self) -> None:
        # No persistent connection is kept; sockets are opened per transaction.
        pass

    def close(self) -> None:
        self._last_response = None

    def is_open(self) -> bool:
        return True

    def write(self, data: bytes) -> None:
        """Send *data* to the guardian and cache its length-prefixed response."""
        self._last_response = self._transact(data)

    def read(self, n: int, timeout_s: Optional[float] = None) -> bytes:
        """Return the response cached by the most recent ``write``.

        The *n* argument is ignored; the guardian already decided how many
        bytes were in the response.
        """
        response = self._last_response
        self._last_response = None
        return response if response is not None else b""

    def _recvall(self, sock: socket.socket, n: int) -> bytes:
        chunks = []
        received = 0
        while received < n:
            chunk = sock.recv(n - received)
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _transact(self, request: bytes) -> bytes:
        sock: Optional[socket.socket] = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self._timeout_s)
            sock.connect(self._socket_path)
            sock.sendall(request)
            length_bytes = self._recvall(sock, 4)
            if len(length_bytes) < 4:
                return b""
            length = struct.unpack(">I", length_bytes)[0]
            return self._recvall(sock, length)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def transact(self, request: bytes, expected_min_len: int, timeout_s: float) -> bytes:
        self.write(request)
        return self.read(expected_min_len, timeout_s=timeout_s)
