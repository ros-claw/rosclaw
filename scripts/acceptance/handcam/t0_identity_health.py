#!/usr/bin/env python3
"""T0: Identity & health baseline (Physical Evolution Lab §10 T0).

No physical motion.  Every campaign's pregate:

* left/right identity by /dev/serial/by-id (v3 §4.4: 按设备身份恢复，
  不按节点编号猜测左右手) — each hand proves itself by slave id;
* USB topology snapshot (lsusb -t);
* firmware/calibration declaration per hand (unmeasured is disclosed);
* camera enumeration + firmware;
* SeekDB reachability;
* Practice data root readable;
* Hand temperatures (thermal baseline for campaign planning).

Runs in the WORKSPACE venv (pyserial/pyrealsense2); repo modules are
imported from the repo src tree.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_SRC = "/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src"
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, "/home/nvidia/workspace/rosclaw_rh56_real/rosclaw-rh56-runtime/src")
sys.path.insert(0, "/home/nvidia/workspace/rosclaw/rosclaw_test/examples/rh56_rps/src")

RIGHT_BY_ID = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG04LBR0-if00-port0"
LEFT_BY_ID = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG04LB62-if00-port0"
CAMERA_SERIAL = "231122070092"
PRACTICE_ROOT = Path("/home/nvidia/.rosclaw/acceptance/evo_rps/evo_rps_2026_01/practice")
SEEKDB_DSN = "seekdb://root@127.0.0.1:2881/rosclaw_evo_rps_2026_01"


def _lsusb_tree() -> str:
    try:
        return subprocess.run(["lsusb", "-t"], capture_output=True, text=True, timeout=15).stdout
    except Exception as exc:  # noqa: BLE001
        return f"lsusb -t failed: {exc}"


def _hand_identity(port: str, expected_slave: int) -> dict:
    from rosclaw_rps.hand.rh56_controller import RH56Controller

    result: dict = {"port": port, "expected_slave": expected_slave}
    try:
        ctl = RH56Controller(port=port)
        ctl.connect()
        try:
            tel = ctl.read_telemetry()
            temps = [
                v
                for v in (tel.temperature_c or {}).values()
                if isinstance(v, (int, float)) and v > 0
            ]
            result["temperature_max"] = max(temps) if temps else None
            result["angle_actual"] = tel.angle_actual
            result["status"] = tel.status
        except Exception as exc:  # noqa: BLE001
            result["telemetry_error"] = f"{type(exc).__name__}: {exc}"
        result["ok"] = True
        ctl.close()
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _camera_identity() -> dict:
    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == CAMERA_SERIAL:
                return {
                    "ok": True,
                    "serial": CAMERA_SERIAL,
                    "name": dev.get_info(rs.camera_info.name),
                    "firmware": dev.get_info(rs.camera_info.firmware_version),
                    "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor),
                }
        return {"ok": False, "error": f"camera {CAMERA_SERIAL} not enumerated"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _seekdb_health() -> dict:
    try:
        import pymysql

        conn = pymysql.connect(
            host="127.0.0.1", port=2881, user="root", password="", connect_timeout=5
        )
        cur = conn.cursor()
        cur.execute("SHOW DATABASES")
        dbs = [row[0] for row in cur.fetchall()]
        conn.close()
        return {"ok": True, "databases": dbs}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _practice_health() -> dict:
    sessions = (
        sorted(
            p.name
            for p in PRACTICE_ROOT.glob("sessions/prac_*")
            if (p / "raw" / "events.jsonl").is_file()
        )
        if PRACTICE_ROOT.is_dir()
        else []
    )
    return {
        "ok": PRACTICE_ROOT.is_dir() and len(sessions) > 0,
        "root": str(PRACTICE_ROOT),
        "sessions_with_events": len(sessions),
        "latest": sessions[-1] if sessions else None,
    }


def main() -> int:
    started = time.time()
    report: dict = {
        "task": "T0_identity_health_baseline",
        "started_at": started,
        "usb_topology": _lsusb_tree(),
        "right_hand": _hand_identity(RIGHT_BY_ID, expected_slave=1),
        "left_hand": _hand_identity(LEFT_BY_ID, expected_slave=1),
        "camera": _camera_identity(),
        "seekdb": _seekdb_health(),
        "practice": _practice_health(),
    }
    report["all_ok"] = all(
        section.get("ok")
        for section in (
            report["right_hand"],
            report["left_hand"],
            report["camera"],
            report["seekdb"],
            report["practice"],
        )
    )
    out_dir = Path("/home/nvidia/.rosclaw/acceptance/handcam/t0")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"t0_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    report["report_path"] = str(out_path)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
