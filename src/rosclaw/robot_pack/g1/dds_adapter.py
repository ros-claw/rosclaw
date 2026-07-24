"""Canonical Unitree HG DDS loopback validation for the official G1 simulator."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes, hash_json


@dataclass(frozen=True)
class G1DDSLoopbackReceipt:
    dds_domain: int
    dds_interface: str
    lowcmd_topic: str
    lowstate_topic: str
    commands_published: int
    states_received: int
    physics_steps: int
    actuator_count: int
    finite_state: bool
    imu_observed: bool
    command_feedback_observed: bool
    worker_receipt_hash: str
    worker_stderr: str
    real_hardware_opened: bool = False
    schema_version: str = "rosclaw.g1.dds_loopback_receipt.v1"

    @property
    def passed(self) -> bool:
        return bool(
            self.dds_domain != 0
            and self.dds_interface == "lo"
            and self.lowcmd_topic == "rt/lowcmd"
            and self.lowstate_topic == "rt/lowstate"
            and self.commands_published > 0
            and self.states_received > 0
            and self.physics_steps > 0
            and self.actuator_count == 29
            and self.finite_state
            and self.imu_observed
            and self.command_feedback_observed
            and not self.real_hardware_opened
        )

    @property
    def receipt_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def run_unitree_dds_loopback(
    *,
    unitree_mujoco_root: Path,
    output_dir: Path,
    source_checkout: Path,
    domain_id: int = 73,
    timeout_sec: float = 12.0,
) -> G1DDSLoopbackReceipt:
    if domain_id == 0 or not 1 <= domain_id <= 230:
        raise ValueError("DDS loopback requires an isolated nonzero domain")
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("DDS evidence output must be outside source checkout")
    model_path = unitree_mujoco_root.expanduser().resolve() / "unitree_robots/g1/scene_29dof.xml"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    root.mkdir(parents=True, exist_ok=False)
    ready = root / "worker.ready"
    stop = root / "worker.stop"
    worker_receipt = root / "worker-receipt.json"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rosclaw.robot_pack.g1.dds_sim_worker",
            "--model",
            str(model_path),
            "--domain",
            str(domain_id),
            "--ready",
            str(ready),
            "--stop",
            str(stop),
            "--receipt",
            str(worker_receipt),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + timeout_sec
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if not ready.exists():
        _terminate(process)
        stdout, stderr = process.communicate()
        raise RuntimeError(f"DDS worker did not become ready: {stdout}\n{stderr}")

    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    states: list[LowState_] = []

    def receive(state: LowState_) -> None:
        states.append(state)

    ChannelFactoryInitialize(domain_id, "lo")
    state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    state_subscriber.Init(receive, 16)
    command_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    command_publisher.Init()
    command = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()
    published = 0
    while time.monotonic() < deadline and (len(states) < 30 or published < 50):
        if states:
            latest = states[-1]
            for index in range(29):
                command.motor_cmd[index].mode = 1
                command.motor_cmd[index].q = float(latest.motor_state[index].q)
                command.motor_cmd[index].dq = 0.0
                command.motor_cmd[index].tau = 0.0
                command.motor_cmd[index].kp = 5.0
                command.motor_cmd[index].kd = 0.5
        command.crc = crc.Crc(command)
        if command_publisher.Write(command, timeout=0.2):
            published += 1
        time.sleep(0.01)
    stop.write_text("stop\n", encoding="utf-8")
    state_subscriber.Close()
    command_publisher.Close()
    try:
        stdout, stderr = process.communicate(timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        _terminate(process)
        stdout, stderr = process.communicate()
    if process.returncode != 0 or not worker_receipt.is_file():
        raise RuntimeError(f"DDS worker failed ({process.returncode}): {stdout}\n{stderr}")
    worker = json.loads(worker_receipt.read_text(encoding="utf-8"))
    finite_feedback = bool(states) and all(
        math.isfinite(float(state.motor_state[index].q))
        and math.isfinite(float(state.motor_state[index].dq))
        for state in states
        for index in range(29)
    )
    imu_observed = bool(states) and all(
        math.isfinite(float(value))
        for value in (
            *states[-1].imu_state.quaternion,
            *states[-1].imu_state.gyroscope,
            *states[-1].imu_state.accelerometer,
        )
    )
    receipt = G1DDSLoopbackReceipt(
        dds_domain=domain_id,
        dds_interface="lo",
        lowcmd_topic="rt/lowcmd",
        lowstate_topic="rt/lowstate",
        commands_published=published,
        states_received=len(states),
        physics_steps=int(worker["physics_steps"]),
        actuator_count=int(worker["actuator_count"]),
        finite_state=finite_feedback and bool(worker["finite_state"]),
        imu_observed=imu_observed,
        command_feedback_observed=bool(worker["commands_received"] > 0),
        worker_receipt_hash=hash_bytes(worker_receipt.read_bytes()),
        worker_stderr=stderr[-2000:],
    )
    (root / "dds-loopback-receipt.json").write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


__all__ = ["G1DDSLoopbackReceipt", "run_unitree_dds_loopback"]
