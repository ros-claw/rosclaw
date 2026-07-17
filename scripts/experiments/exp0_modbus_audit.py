#!/usr/bin/env python3
"""Experiment 0 audit: real SerialModbusTransport bring-up + fixture capture.

Runs the frame-layer audit on the physical RH56 and captures golden fixtures
into ``tests/fixtures/rh56_modbus/`` so frame parsing can regress without
hardware.

Usage (from repo root, venv):
    python scripts/experiments/exp0_modbus_audit.py --device /dev/ttyUSB1 --slave-id 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rosclaw.body.rh56 import modbus
from rosclaw.body.rh56.transport import SerialModbusTransport
from rosclaw.body.rh56.transport_profile import (
    CommandConfig,
    TransportConfig,
    TransportProfile,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "rh56_modbus"


def _profile(device: str, slave_id: int) -> TransportProfile:
    return TransportProfile(
        id="rh56_exp0_audit",
        transport=TransportConfig(
            type="serial_modbus_rtu", device=device, baudrate=115200, slave_id=slave_id
        ),
        command=CommandConfig(
            actuator_count=6,
            position_range=[0, 1000],
            position_convention={"closed": 0, "open": 1000},
        ),
        action_order=["little", "ring", "middle", "index", "thumb", "thumb_rot"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/ttyUSB1")
    parser.add_argument("--slave-id", type=int, default=1)
    parser.add_argument("--no-move", action="store_true", help="Skip the write_position probe")
    args = parser.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"device": args.device, "slave_id": args.slave_id, "checks": {}}

    # --- fixture 1: hand-crafted frames (builder self-check) ----------------
    read_req = modbus.build_read_holding_registers(args.slave_id, modbus.Register.ANGLE_ACT, 6)
    (FIXTURE_DIR / "read_positions_request.bin").write_bytes(read_req)
    set_req = modbus.build_write_multiple_registers(
        args.slave_id, modbus.Register.ANGLE_SET, [500] * 6
    )
    (FIXTURE_DIR / "set_positions_request.bin").write_bytes(set_req)
    # Synthetic exception + crc-failure fixtures (for offline regression).
    exc_frame = bytes([args.slave_id, 0x83, 0x02])
    exc_frame += exc_frame.__class__(modbus.crc16_modbus(exc_frame).to_bytes(2, "little"))
    (FIXTURE_DIR / "exception_response.bin").write_bytes(exc_frame)
    bad = bytearray(read_req)
    bad[-1] ^= 0xFF
    (FIXTURE_DIR / "crc_failure.bin").write_bytes(bytes(bad))

    # --- connect -------------------------------------------------------------
    profile = _profile(args.device, args.slave_id)
    transport = SerialModbusTransport(profile)
    transport.connect()
    report["checks"]["connect"] = True

    # --- identity + register audit -------------------------------------------
    hand_id = transport._read_registers(modbus.Register.HAND_ID, 1)[0]
    voltage = transport._read_registers(modbus.Register.VLTAGE, 1)[0]
    report["checks"]["hand_id"] = {"read": hand_id, "matches_slave": hand_id == args.slave_id}
    report["checks"]["voltage_raw"] = voltage

    # Capture a real read response frame.
    req = modbus.build_read_holding_registers(args.slave_id, modbus.Register.ANGLE_ACT, 6)
    frame = transport._transact(req, is_read=True)
    (FIXTURE_DIR / "read_positions_response.bin").write_bytes(frame)
    report["checks"]["read_response_len"] = len(frame)

    # --- read_state semantics -------------------------------------------------
    t0 = time.monotonic()
    state = transport.read_state()
    latency_ms = (time.monotonic() - t0) * 1000
    report["checks"]["read_state_latency_ms"] = round(latency_ms, 1)
    report["checks"]["position"] = state.position
    report["checks"]["force_g"] = state.force_g
    report["checks"]["current_ma"] = state.current_ma
    report["checks"]["status_bits"] = state.status_bits
    report["checks"]["temperature_c"] = state.temperature_c

    # --- write + read-back (small safe probe unless --no-move) ----------------
    if not args.no_move:
        target = [p if p < 950 else p - 50 for p in state.position]  # tiny flex, ≤50 raw
        ok = transport.write_position(target, speed=200, force_limit=200)
        time.sleep(0.4)
        after = transport.read_state()
        report["checks"]["write_position"] = {
            "ok": ok,
            "delivery": transport.last_command_delivery,
            "target": target,
            "actual_after": after.position,
        }
        # Restore original pose.
        restore_ok = transport.write_position(state.position, speed=200, force_limit=200)
        report["checks"]["restore"] = {
            "ok": restore_ok,
            "delivery": transport.last_command_delivery,
        }

    # --- exception frame audit -------------------------------------------------
    try:
        transport._read_registers(0x0000, 1)  # illegal address per manual map
        report["checks"]["exception_probe"] = "no_exception (address accepted)"
    except modbus.ModbusExceptionError as exc:
        report["checks"]["exception_probe"] = f"modbus_exception: {exc}"
    except Exception as exc:  # noqa: BLE001
        report["checks"]["exception_probe"] = f"other: {type(exc).__name__}: {exc}"

    # --- reopen policy ----------------------------------------------------------
    transport.close()
    transport.connect()
    state2 = transport.read_state()
    report["checks"]["reopen_read_ok"] = len(state2.position) == 6

    transport.close()

    out = Path("/tmp/exp0_modbus_audit.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nfixtures -> {FIXTURE_DIR}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
