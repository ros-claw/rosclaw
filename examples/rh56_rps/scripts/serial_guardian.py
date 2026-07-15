#!/usr/bin/env python3
"""Serial guardian daemon for CH340 USB/RS485 adapters.

Keeps real serial ports open forever and exposes each one on a Unix domain
socket.  Clients (the RPS demo) connect, send a Modbus request, and receive a
length-prefixed response.  Because the physical serial device is never closed,
the CH340 close/reopen kernel -110 / Input/output error bug is avoided.

Usage (after reseating the CH340 adapters):

    PYTHONPATH=src:/path/to/rosclaw-rh56-runtime/src \
        python3 scripts/serial_guardian.py --socket-dir /tmp/rh56_guardian

Then run the demo with:

    RH56_GUARDIAN=1 ./run_rosclaw_rps.sh --mode full --auto --headless
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import serial

from rosclaw_rh56.protocol.inspire_protocol import (
    build_read_holding_registers,
    parse_read_response,
    verify_modbus_response,
)

HAND_ID_REGISTER = 0x03E8
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT_S = 0.3
MAX_RESPONSE_BYTES = 64
logger = logging.getLogger("rh56_serial_guardian")


class _GuardianPort:
    def __init__(
        self,
        port: str,
        device_id: int,
        socket_path: str,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        existing_serial: Optional[serial.Serial] = None,
    ):
        self.port = port
        self.device_id = device_id
        self.socket_path = socket_path
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self._existing_serial = existing_serial
        self._ser: Optional[serial.Serial] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._existing_serial is not None:
            self._ser = self._existing_serial
            self._clear_hupcl()
            logger.info("Reusing already-open %s for device_id=%s", self.port, self.device_id)
        else:
            self._open_serial()
        self._ensure_socket_removed()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        logger.info(
            "Guardian serving device_id=%s port=%s on %s",
            self.device_id,
            self.port,
            self.socket_path,
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._ensure_socket_removed()
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def _open_serial(self) -> None:
        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout_s,
        )
        # Clear HUPCL so DTR stays high even if the daemon happens to close.
        self._clear_hupcl()
        logger.info("Opened %s for device_id=%s", self.port, self.device_id)

    def _clear_hupcl(self) -> None:
        import termios

        if self._ser is None or self._ser.fd is None:
            return
        try:
            attrs = termios.tcgetattr(self._ser.fd)
            attrs[2] &= ~termios.HUPCL
            termios.tcsetattr(self._ser.fd, termios.TCSANOW, attrs)
        except Exception:
            pass

    def _ensure_socket_removed(self) -> None:
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.debug("Could not remove socket %s: %s", self.socket_path, exc)

    def _serve(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(self.socket_path)
            os.chmod(self.socket_path, 0o666)
            sock.listen(4)
            while not self._stop_event.is_set():
                sock.settimeout(1.0)
                try:
                    client, _ = sock.accept()
                except socket.timeout:
                    continue
                client.settimeout(self.timeout_s)
                self._handle_client(client)
        finally:
            try:
                sock.close()
            except Exception:
                pass
            self._ensure_socket_removed()

    def _handle_client(self, client: socket.socket) -> None:
        try:
            while not self._stop_event.is_set():
                request = self._read_request(client)
                if not request:
                    break
                response = self._serial_transact(request)
                framed = struct.pack(">I", len(response)) + response
                client.sendall(framed)
        except Exception as exc:
            logger.debug("Client handler error for device_id=%s: %s", self.device_id, exc)
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _read_request(self, client: socket.socket) -> bytes:
        # Modbus RTU requests from the RH56 protocol are short (<= 64 bytes).
        chunks = []
        client.settimeout(0.5)
        deadline = time.time() + 0.5
        while time.time() < deadline:
            try:
                chunk = client.recv(64)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
            if sum(len(c) for c in chunks) >= 8:
                break
            time.sleep(0.005)
        return b"".join(chunks)

    def _serial_transact(self, request: bytes) -> bytes:
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError(f"Serial port {self.port} is not open")
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self._ser.write(request)
        self._ser.flush()
        # Read response with a short inter-byte gap heuristic.
        deadline = time.time() + self.timeout_s
        response = bytearray()
        while time.time() < deadline and len(response) < MAX_RESPONSE_BYTES:
            waiting = self._ser.in_waiting
            if waiting:
                chunk = self._ser.read(min(waiting, MAX_RESPONSE_BYTES - len(response)))
                response.extend(chunk)
                deadline = time.time() + 0.03
            else:
                time.sleep(0.005)
        return bytes(response)


def discover_hands(
    port_glob: str = "/dev/ttyUSB*",
    slave_ids: List[int] = (1, 2),
    baudrate: int = DEFAULT_BAUDRATE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[Dict[int, str], Dict[int, serial.Serial]]:
    """Discover which serial port each RH56 hand is attached to.

    Each candidate port is opened once and kept open.  The returned serial
    objects can be handed off to the guardian threads so the physical ports are
    never closed and reopened.
    """
    ports = sorted(glob.glob(port_glob))
    if not ports:
        raise RuntimeError(f"No serial ports matched by {port_glob}")

    mapping: Dict[int, str] = {}
    serials: Dict[int, serial.Serial] = {}
    for port in ports:
        ser: Optional[serial.Serial] = None
        try:
            ser = serial.Serial(port, baudrate=baudrate, timeout=timeout_s)
            time.sleep(0.05)
            for slave_id in slave_ids:
                if slave_id in mapping:
                    continue
                request = build_read_holding_registers(slave_id, HAND_ID_REGISTER, quantity=1)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                ser.write(request)
                time.sleep(0.05)
                response = ser.read(64)
                if verify_modbus_response(response, slave_id, function_code=0x03):
                    values = parse_read_response(response)
                    if values and values[0] == slave_id:
                        mapping[slave_id] = port
                        serials[slave_id] = ser
                        logger.info("Discovered hand slave_id=%s on %s", slave_id, port)
        except Exception as exc:
            logger.warning("Could not probe %s: %s", port, exc)
        finally:
            # Only close ports where we did not find a hand.
            if ser is not None and ser.is_open and port not in mapping.values():
                try:
                    ser.close()
                except Exception:
                    pass
    return mapping, serials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-dir", default="/tmp/rh56_guardian")
    parser.add_argument("--port-glob", default="/dev/ttyUSB*")
    parser.add_argument("--slave-ids", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    os.makedirs(args.socket_dir, exist_ok=True)

    mapping, serials = discover_hands(
        port_glob=args.port_glob,
        slave_ids=args.slave_ids,
        baudrate=args.baudrate,
        timeout_s=args.timeout,
    )
    if not mapping:
        logger.error("No RH56 hands found.")
        return 1

    guardians: List[_GuardianPort] = []
    for device_id, port in mapping.items():
        socket_path = os.path.join(args.socket_dir, f"{device_id}.sock")
        gp = _GuardianPort(
            port=port,
            device_id=device_id,
            socket_path=socket_path,
            baudrate=args.baudrate,
            timeout_s=args.timeout,
            existing_serial=serials.get(device_id),
        )
        gp.start()
        guardians.append(gp)

    logger.info("Guardian running. Press Ctrl-C to stop.")
    stop_event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _s, _f: stop_event.set())
    while not stop_event.is_set():
        time.sleep(0.5)

    logger.info("Shutting down guardian...")
    for gp in guardians:
        gp.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
