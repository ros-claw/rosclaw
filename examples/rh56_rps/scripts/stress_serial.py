#!/usr/bin/env python3
"""Stress-test CH340 open/close cycles.

Opens each /dev/ttyUSB* port, reads the HAND_ID register once, and closes it.
Reports after how many cycles the kernel returns -110 / Input/output error.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import serial
import termios

from rosclaw_rh56.protocol.inspire_protocol import (
    build_read_holding_registers,
    verify_modbus_response,
    parse_read_response,
)

HAND_ID_REGISTER = 0x03E8


def _clear_hupcl(ser: serial.Serial) -> None:
    """Keep DTR asserted on close to avoid CH340 reset/reopen failures."""
    if ser.fd is None:
        return
    try:
        attrs = termios.tcgetattr(ser.fd)
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(ser.fd, termios.TCSANOW, attrs)
    except Exception:
        pass


def probe_port(port: str, slave_id: int, baudrate: int, timeout_s: float) -> bool:
    request = build_read_holding_registers(slave_id, HAND_ID_REGISTER, quantity=1)
    try:
        with serial.Serial(port, baudrate=baudrate, timeout=timeout_s) as ser:
            _clear_hupcl(ser)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            ser.write(request)
            time.sleep(0.05)
            response = ser.read(64)
    except Exception as exc:
        print(f"  {port} slave={slave_id}: OPEN/PROBE FAILED: {exc}")
        return False

    if not verify_modbus_response(response, slave_id, function_code=0x03):
        print(f"  {port} slave={slave_id}: bad modbus response")
        return False
    values = parse_read_response(response)
    if values and values[0] == slave_id:
        print(f"  {port} slave={slave_id}: OK")
        return True
    print(f"  {port} slave={slave_id}: unexpected response id={values}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="/dev/ttyUSB*")
    parser.add_argument("--left-id", type=int, default=1)
    parser.add_argument("--right-id", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.3)
    args = parser.parse_args()

    import glob as _glob
    ports = sorted(_glob.glob(args.glob))
    if not ports:
        print(f"No ports matched {args.glob}")
        return 1

    slave_ids = [args.left_id, args.right_id]
    for cycle in range(1, args.cycles + 1):
        print(f"Cycle {cycle}/{args.cycles}")
        ok = True
        for port in ports:
            for slave_id in slave_ids:
                if not probe_port(port, slave_id, args.baudrate, args.timeout):
                    ok = False
        if not ok:
            print(f"FAILED at cycle {cycle}")
            return 1
        time.sleep(0.1)

    print(f"All {args.cycles} cycles passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
