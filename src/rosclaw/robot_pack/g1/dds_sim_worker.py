"""Bounded official Unitree G1 MuJoCo process behind isolated DDS loopback."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--domain", type=int, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=4000)
    args = parser.parse_args()
    if args.domain == 0:
        raise SystemExit("DDS simulator refuses real-robot domain 0")

    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if model.nu != 29:
        raise SystemExit(f"DDS G1 worker requires 29 actuators, got {model.nu}")
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    command_count = 0
    state_count = 0
    last_command = None
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64) * 0.5

    def receive(command: LowCmd_) -> None:
        nonlocal command_count, last_command
        command_count += 1
        last_command = command

    ChannelFactoryInitialize(args.domain, "lo")
    state_publisher = ChannelPublisher("rt/lowstate", LowState_)
    state_publisher.Init()
    command_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    command_subscriber.Init(receive, 8)
    state = unitree_hg_msg_dds__LowState_()
    args.ready.write_text("ready\n", encoding="utf-8")
    finite = True
    peak_torque_scale = 0.0
    started = time.monotonic()
    steps = 0
    for steps in range(1, args.max_steps + 1):
        if last_command is not None:
            torque = np.asarray(
                [
                    float(last_command.motor_cmd[index].tau)
                    + float(last_command.motor_cmd[index].kp)
                    * (float(last_command.motor_cmd[index].q) - float(data.sensordata[index]))
                    + float(last_command.motor_cmd[index].kd)
                    * (
                        float(last_command.motor_cmd[index].dq)
                        - float(data.sensordata[index + model.nu])
                    )
                    for index in range(29)
                ],
                dtype=np.float64,
            )
            peak_torque_scale = max(
                peak_torque_scale,
                float(np.max(np.abs(torque) / np.asarray(G1_HARD_TORQUE_LIMITS))),
            )
            data.ctrl[:] = np.clip(torque, -limits, limits)
        mujoco.mj_step(model, data)
        finite = finite and bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.ctrl))
        )
        for index in range(29):
            state.motor_state[index].q = float(data.sensordata[index])
            state.motor_state[index].dq = float(data.sensordata[index + model.nu])
            state.motor_state[index].tau_est = float(data.sensordata[index + 2 * model.nu])
        _fill_imu(model, data, state)
        state_publisher.Write(state)
        state_count += 1
        if args.stop.exists() and steps >= 100:
            break
        if not finite:
            break
        deadline = started + steps * model.opt.timestep
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    command_subscriber.Close()
    state_publisher.Close()
    receipt = {
        "schema_version": "rosclaw.g1.dds_sim_worker.v1",
        "dds_domain": args.domain,
        "dds_interface": "lo",
        "topics": ["rt/lowcmd", "rt/lowstate"],
        "model": str(args.model.resolve()),
        "actuator_count": int(model.nu),
        "physics_steps": steps,
        "commands_received": command_count,
        "states_published": state_count,
        "finite_state": finite,
        "peak_torque_scale": peak_torque_scale,
        "real_hardware_opened": False,
    }
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if finite and command_count > 0 and state_count > 0 else 2


def _fill_imu(model: mujoco.MjModel, data: mujoco.MjData, state: Any) -> None:
    names = ("imu_quat", "imu_gyro", "imu_acc")
    sensor_values: dict[str, np.ndarray] = {}
    for name in names:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            continue
        address = int(model.sensor_adr[sensor_id])
        dimension = int(model.sensor_dim[sensor_id])
        sensor_values[name] = data.sensordata[address : address + dimension]
    quaternion = sensor_values.get("imu_quat", np.asarray((1.0, 0.0, 0.0, 0.0)))
    gyroscope = sensor_values.get("imu_gyro", np.zeros(3))
    accelerometer = sensor_values.get("imu_acc", np.zeros(3))
    if not all(math.isfinite(float(value)) for value in quaternion):
        quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
    for index in range(4):
        state.imu_state.quaternion[index] = float(quaternion[index])
    for index in range(3):
        state.imu_state.gyroscope[index] = float(gyroscope[index])
        state.imu_state.accelerometer[index] = float(accelerometer[index])


if __name__ == "__main__":
    raise SystemExit(main())
