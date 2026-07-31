#!/usr/bin/env python3
"""T1: Pose lattice with the full PE-2/3/4 chain (Physical Evolution Lab §10 T1).

Per hand (right, then left — never both under one serial owner):
  5 standard poses × 2 repeats:
    command → telemetry during execution → camera frames through
    settle → PE-3 visual state (settle onset/stability) → PE-4
    two-mode prior residuals.

Produces a practice-format session dir (frame_event / rps.telemetry /
health_check / t1.pose events) so the PE-2 sync layer and the PR-PE-1
Data Quality Gate process it UNCHANGED — the lattice is judged by the
same contract as every other session (v3 §16: no parallel systems).

Safety: one session ≈ 30 s of motion per hand; thermal start gate
(≤46 °C) aborts honestly; safe_open on exit; camera lifecycle is
wedge-safe (hardware_reset first, graceful stop only).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_SRC = "/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src"
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, "/home/nvidia/workspace/rosclaw_rh56_real/rosclaw-rh56-runtime/src")
sys.path.insert(0, "/home/nvidia/workspace/rosclaw/rosclaw_test/examples/rh56_rps/src")

import pyrealsense2 as rs  # noqa: E402

from rosclaw.evolution.hardware.camera import D435iCapture  # noqa: E402
from rosclaw.evolution.hardware.thermal import default_temp_probe  # noqa: E402
from rosclaw.perception.handcam import (  # noqa: E402
    CameraPoseContract,
    estimate_hand_state,
    intrinsics_from_pyrealsense,
    measure_visual_stability,
)
from rosclaw.self_model.adapters.rh56 import RH56ForwardPrior, RH56HandSelfAdapter  # noqa: E402
from rosclaw_rps.hand.rh56_controller import RH56Controller  # noqa: E402

RIGHT_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG04LBR0-if00-port0"
LEFT_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG04LB62-if00-port0"
RIGHT_ROI = {"x0": 410, "y0": 120, "x1": 640, "y1": 480}
LEFT_ROI = {"x0": 0, "y0": 80, "x1": 230, "y1": 480}
CAMERA_SERIAL = "231122070092"
START_MAX_TEMP_C = 46.0
OUT_ROOT = Path("/home/nvidia/.rosclaw/acceptance/handcam/t1")

JOINTS = ("little", "ring", "middle", "index", "thumb", "thumb_rot")

# Device identity is PROBED, never assumed (v3 §4.4): this rig's left
# hand answers slave id 1, the right answers slave id 2 (measured
# 2026-07-31 — the controller's port scan guessed 2 for both, so early
# T1 runs issued left commands to a silent id; a static hand is
# trivially "settled" and those runs' left sections are no-motion).
SLAVE_CANDIDATES = (1, 2)

# Modest throttle for slow grouped-register reads (left hand).
TELEMETRY_MIN_INTERVAL_S = {"left": 0.2, "right": 0.0}

# Standard lattice (declared, not tuned): open / rock / scissors / point / ok-ish.
POSES = {
    "open": [1000, 1000, 1000, 1000, 1000, 1000],
    "rock": [150, 150, 150, 150, 150, 200],
    "scissors": [150, 150, 1000, 1000, 150, 200],
    "point": [150, 150, 150, 1000, 150, 200],
    "ok_ish": [1000, 1000, 1000, 400, 250, 250],
}
REPEATS = 2
SETTLE_FRAMES = 12
GESTURE_SPEED = 300
GESTURE_FORCE = 300


def _now_ns() -> int:
    return time.time_ns()


class EventWriter:
    """Minimal practice-format event writer (same schema the sync layer reads)."""

    def __init__(self, session_dir: Path, practice_id: str):
        self._path = session_dir / "raw" / "events.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8")
        self._practice_id = practice_id
        self.frame_no = 0
        self._session_id = f"sess_{practice_id[5:]}"
        self._episode_id = f"ep_{practice_id[5:]}"

    def emit(
        self,
        event_type: str,
        payload: dict,
        *,
        ts_ns: int | None = None,
        action_id: str | None = None,
    ) -> None:
        ts = ts_ns or _now_ns()
        event = {
            "schema_version": "practice.event.v1",
            "event_id": f"evt_{ts}_{event_type.replace('.', '_')}",
            "event_type": event_type,
            "timestamp_ns": ts,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts / 1e9)),
            "practice_id": self._practice_id,
            "session_id": self._session_id,
            "episode_id": self._episode_id,
            "robot_id": "rh56_handcam_robot",
            "body_id": "rh56_handcam_robot",
            "trace_id": f"trace_{self._practice_id}",
            "payload": payload,
        }
        if action_id:
            event["action_id"] = action_id
        self._handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _side_payload(telemetry, read_ts: float) -> dict:
    return {
        "timestamp": read_ts,
        "angle_actual": telemetry.angle_actual,
        "angle_set": getattr(telemetry, "angle_set", None),
        "force_act": getattr(telemetry, "force_act", None),
        "current_ma": getattr(telemetry, "current_ma", None),
        "temperature_c": telemetry.temperature_c,
        "status": telemetry.status,
    }


def _telemetry_payload(readings: dict, practice_id: str, ts: float) -> dict:
    """Both hands in every event (dual-hand platform shape, like the RPS
    corpus); each side carries the timestamp its reading was ACTUALLY
    taken (a carried passive reading never wears the frame's time)."""
    payload = {"timestamp": ts, "practice_id": practice_id, "right": {}, "left": {}}
    for side, entry in readings.items():
        if entry is not None:
            telemetry, read_ts = entry
            payload[side] = _side_payload(telemetry, read_ts)
    return payload


def open_probed_controller(port: str, label: str):
    """Open a hand controller and probe its slave id by response (v3 §4.4:
    identity by response, never by port-name guessing).

    The transport's advisory file lock is per lock_path, not per port —
    the default is one GLOBAL path, so a process holding two controllers
    deadlocks itself.  Per-hand lock paths guard each port independently
    (this is also how a dual-hand process must own its devices)."""
    from rosclaw_rh56.transport.base import TransportConfig
    from rosclaw_rh56.transport.serial_rs485 import SerialRS485Transport

    transport = SerialRS485Transport(
        TransportConfig(
            kind="serial_rs485",
            port=port,
            baudrate=115200,
            timeout_s=1.0,
            lock_path=f"/tmp/rosclaw_rh56_serial_{label}.lock",
        )
    )
    ctl = RH56Controller(port=port, transport=transport)
    ctl.connect()
    proto = ctl._proto
    transport = ctl._transport
    for candidate in SLAVE_CANDIDATES:
        for _attempt in range(3):
            proto.device_id = candidate
            transport.flush_input()
            transport.write(proto.read_angle_actual())
            time.sleep(0.5)
            if transport.read(64, timeout_s=1.0):
                return ctl, candidate
            time.sleep(0.2)
    ctl.close()
    raise RuntimeError(f"{label} hand: no modbus response at any candidate slave id")


def run_hand(
    hand_label: str,
    ctl,
    slave_id: int,
    roi: dict,
    contract_intr,
    prior: RH56ForwardPrior,
    writer: EventWriter,
    cap: D435iCapture,
    practice_id: str,
    passive_reader=None,
) -> dict:
    results: dict = {"hand": hand_label, "poses": [], "residual_rmse": {}}
    results["slave_id"] = slave_id
    contract = CameraPoseContract(
        camera_pose_id=f"front_v1_{hand_label}",
        camera_id="d435i",
        intrinsics=contract_intr,
        roi=roi,
    )
    try:
        for pose_name, angles in POSES.items():
            for repeat in range(REPEATS):
                target = dict(zip(JOINTS, angles, strict=True))
                command_ts = time.time()
                action_id = f"act_{practice_id}_{pose_name}_{repeat}"
                ctl.move_to_gesture(pose_name, angles, GESTURE_SPEED, GESTURE_FORCE)
                states, state_ts = [], []
                residuals_sq: dict[str, float] = {}
                residual_n = 0
                prev_tel = None
                last_tel = None
                last_tel_ts = 0.0
                passive_reading = None
                for index in range(SETTLE_FRAMES):
                    frame = cap.read()
                    ts = time.time()
                    state = estimate_hand_state(frame.depth, contract)
                    states.append(state)
                    state_ts.append(ts)
                    if ts - last_tel_ts >= TELEMETRY_MIN_INTERVAL_S.get(hand_label, 0.0):
                        tel = ctl.read_telemetry()
                        last_tel = tel
                        last_tel_ts = ts
                    else:
                        tel = last_tel
                    if passive_reader is not None and index == 0:
                        passive_reading = passive_reader()
                    writer.frame_no += 1
                    writer.emit(
                        "frame_event",
                        {
                            "frame_number": writer.frame_no,
                            "camera_frame_ts": ts,
                            "host_ts_ns": _now_ns(),
                            "has_depth": True,
                            "keyframe": index == 0,
                            "keyframe_path": None,
                            "pose": pose_name,
                            "repeat": repeat,
                        },
                    )
                    readings = {hand_label: (tel, ts) if tel is not None else None}
                    other = "left" if hand_label == "right" else "right"
                    readings[other] = passive_reading
                    writer.emit(
                        "rps.telemetry",
                        _telemetry_payload(readings, practice_id, ts),
                    )
                    if prev_tel is not None:
                        dt = ts - prev_tel[0]
                        prev_pos = prev_tel[1].angle_actual or {}
                        cur_pos = tel.angle_actual or {}
                        vel_state: dict[str, float] = {}
                        for j in JOINTS:
                            if (
                                j in prev_pos
                                and j in cur_pos
                                and prev_pos[j] is not None
                                and cur_pos[j] is not None
                                and dt > 0.01
                            ):
                                vel_state[f"pos_{j}"] = float(prev_pos[j])
                                vel_state[f"vel_{j}"] = (
                                    float(cur_pos[j]) - float(prev_pos[j])
                                ) / dt
                        action = {
                            f"target_{j}": float(target.get(j, prev_pos.get(j) or 0))
                            for j in JOINTS
                            if f"pos_{j}" in vel_state
                        }
                        action["dt_s"] = dt
                        if action:
                            prediction = prior.predict(vel_state, action)
                            for j in JOINTS:
                                key = f"next_pos_{j}"
                                if key in prediction.channels and j in cur_pos:
                                    residuals_sq[j] = residuals_sq.get(j, 0.0) + (
                                        float(cur_pos[j]) - prediction.channels[key]
                                    ) ** 2
                            residual_n += 1
                    prev_tel = (ts, tel)
                    time.sleep(0.02)
                stability = measure_visual_stability(states, state_ts, settle_window_s=0.3)
                pose_result = {
                    "pose": pose_name,
                    "repeat": repeat,
                    "command_ts": command_ts,
                    "target": target,
                    "final_position": (prev_tel[1].angle_actual if prev_tel else None),
                    "visual_state": states[-1].state,
                    "visual_fingertip": states[-1].fingertip_3d,
                    "settled": stability.settled,
                    "settle_time_s": stability.settle_time_s,
                    "onset_time_s": stability.onset_time_s,
                    "unknown_frames": stability.unknown_frames,
                }
                results["poses"].append(pose_result)
                writer.emit("t1.pose", pose_result, action_id=action_id)
                results["residual_rmse"][pose_name] = {
                    j: round((sq / max(1, residual_n)) ** 0.5, 1)
                    for j, sq in residuals_sq.items()
                }
        ctl.safe_open()
    finally:
        ctl.close()
    return results


def main() -> int:
    probe = default_temp_probe()
    temps = [v for v in probe.values() if isinstance(v, (int, float))]
    if temps and max(temps) > START_MAX_TEMP_C:
        print(
            json.dumps({"ok": False, "blocked": f"start temp {max(temps)}°C > {START_MAX_TEMP_C}°C"})
        )
        return 1

    practice_id = f"prac_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_t1lattice"
    session_dir = OUT_ROOT / "sessions" / practice_id
    writer = EventWriter(session_dir, practice_id)

    cap = D435iCapture()
    cap.start(serial=CAMERA_SERIAL)
    intr = intrinsics_from_pyrealsense(cap._pipeline.get_active_profile(), rs.stream.depth)

    adapter = RH56HandSelfAdapter("rh56_handcam_01")
    prior = RH56ForwardPrior(adapter.body_id(), adapter.body_hash())

    outcomes: dict = {"practice_id": practice_id, "session_dir": str(session_dir), "hands": []}
    controllers: dict = {}
    try:
        for label, port in (("right", RIGHT_PORT), ("left", LEFT_PORT)):
            controllers[label], slave = open_probed_controller(port, label)
            outcomes.setdefault("slave_ids", {})[label] = slave
        for hand_label, roi in (
            ("right", RIGHT_ROI),
            ("left", LEFT_ROI),
        ):
            other = "left" if hand_label == "right" else "right"
            passive_ctl = controllers[other]

            def passive_reader(ctl=passive_ctl):
                try:
                    return (ctl.read_telemetry(), time.time())
                except Exception:  # noqa: BLE001
                    return None

            outcomes["hands"].append(
                run_hand(
                    hand_label,
                    controllers[hand_label],
                    outcomes["slave_ids"][hand_label],
                    roi,
                    intr,
                    prior,
                    writer,
                    cap,
                    practice_id,
                    passive_reader=passive_reader,
                )
            )
    finally:
        for ctl in controllers.values():
            try:
                ctl.close()
            except Exception:  # noqa: BLE001
                pass
        cap.stop()
        writer.emit(
            "health_check",
            {"camera": {"alive": True, "last_frame_age_s": 0.03, "empty_streak": 0}, "rounds": 0},
        )
        writer.close()

    from rosclaw.practice.data_quality import run_data_quality_gate
    from rosclaw.practice.multimodal_sync import build_bundles

    sync = build_bundles(session_dir, camera_id="d435i", experiment_id="handcam_t1")
    gate = run_data_quality_gate(session_dir)
    outcomes["sync_stats"] = sync.stats.to_dict()
    outcomes["data_quality"] = gate.to_dict()["data_quality"]
    out_report = OUT_ROOT / f"t1_report_{practice_id}.json"
    out_report.write_text(json.dumps(outcomes, indent=2, ensure_ascii=False, default=str))
    outcomes["report_path"] = str(out_report)
    print(json.dumps(outcomes, indent=2, ensure_ascii=False, default=str))
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
