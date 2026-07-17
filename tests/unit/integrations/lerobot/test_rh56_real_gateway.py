"""REAL-mode RH56 execution through the Runtime ActionGateway (P5-D).

Uses the loopback fake serial (mini device model) so the *real*
SerialModbusTransport (execution_mode="REAL") drives the full pipeline:

    ActionEnvelope → ActionGateway → RH56RealStepExecutor
      → SingleStepExecutor (permit/watchdog/freshness)
      → RH56Executor → device → feedback verify → ExecutionReceipt

Fail-closed pairings (FIXTURE executor on REAL transport and vice versa)
must block without dispatching.
"""

from __future__ import annotations

import struct
import time

from rosclaw.body.execution.rh56_executor import RH56Executor
from rosclaw.body.rh56 import modbus
from rosclaw.body.rh56.transport import MockModbusTransport, SerialModbusTransport
from rosclaw.body.rh56.transport_profile import (
    CommandConfig,
    TransportConfig,
    TransportProfile,
)
from rosclaw.integrations.lerobot.execution import (
    ArmingController,
    FeedbackVerifier,
    PermitManager,
    SingleStepExecutor,
)
from rosclaw.integrations.lerobot.execution.rh56_real_executor import (
    CAPABILITY_ID,
    RH56RealStepExecutor,
)
from rosclaw.kernel.action_gateway import ActionGateway
from rosclaw.kernel.contracts import (
    ActionEnvelope,
    ActionState,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)

HASHES = {
    "policy_contract_hash": "sha256:policy",
    "body_hash": "sha256:body",
    "calibration_hash": "sha256:calibration",
    "mapping_hash": "sha256:mapping",
    "transport_profile_hash": "sha256:profile",
}


def _profile() -> TransportProfile:
    return TransportProfile(
        id="test_profile",
        transport=TransportConfig(
            type="serial_modbus_rtu", device="/dev/null", baudrate=115200, slave_id=1
        ),
        command=CommandConfig(
            actuator_count=6,
            position_range=[0, 1000],
            position_convention={"closed": 0, "open": 1000},
        ),
        action_order=["little", "ring", "middle", "index", "thumb", "thumb_rot"],
    )


class _FakeDevice:
    """Mini register file; ANGLE_ACT snaps to ANGLE_SET on read (instant move)."""

    def __init__(self, slave_id: int = 1):
        self.slave_id = slave_id
        self.registers = {modbus.Register.HAND_ID: [slave_id]}
        for base in (
            modbus.Register.ANGLE_ACT,
            modbus.Register.FORCE_ACT,
            modbus.Register.CURRENT,
            modbus.Register.STATUS,
            modbus.Register.TEMP,
            modbus.Register.ANGLE_SET,
            modbus.Register.SPEED_SET,
            modbus.Register.FORCE_SET,
        ):
            self.registers[base] = [1000] * 6
        self.registers[modbus.Register.FORCE_ACT] = [0] * 6
        self.registers[modbus.Register.CURRENT] = [0] * 6
        self.registers[modbus.Register.STATUS] = [2] * 6
        self.registers[modbus.Register.TEMP] = [35] * 6

    def handle(self, request: bytes) -> bytes:
        slave, fc = request[0], request[1]
        assert slave == self.slave_id
        if fc == 0x03:
            start, qty = struct.unpack(">HH", request[2:6])
            if start == modbus.Register.ANGLE_ACT:
                self.registers[modbus.Register.ANGLE_ACT] = list(
                    self.registers[modbus.Register.ANGLE_SET]
                )
            values = self.registers.get(start, [0] * qty)[:qty]
            body = bytes([slave, 0x03, qty * 2]) + b"".join(
                struct.pack(">H", v & 0xFFFF) for v in values
            )
        elif fc == 0x10:
            start, qty = struct.unpack(">HH", request[2:6])
            data = request[7 : 7 + qty * 2]
            self.registers[start] = [
                struct.unpack(">H", data[i : i + 2])[0] for i in range(0, qty * 2, 2)
            ]
            body = bytes([slave, 0x10]) + request[2:6]
        else:
            body = bytes([slave, fc | 0x80, 0x02])
        return body + struct.pack("<H", modbus.crc16_modbus(body))


class _FakeSerial:
    def __init__(self, device: _FakeDevice):
        self._device = device
        self.is_open = True
        self.timeout = 0.05
        self._rx = bytearray()

    def write(self, data: bytes) -> int:
        self._rx += self._device.handle(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, n: int) -> bytes:
        if not self._rx:
            time.sleep(0.001)
            return b""
        out = bytes(self._rx[:n])
        del self._rx[:n]
        return out

    def reset_input_buffer(self) -> None:
        self._rx.clear()

    def close(self) -> None:
        self.is_open = False

    def open(self) -> None:
        self.is_open = True


def _armed_stack(
    *,
    mode: ExecutionMode = ExecutionMode.REAL,
    transport=None,
    calibration=None,
    max_step_delta_raw: float = 30.0,
):
    profile = _profile()
    if transport is None:
        transport = SerialModbusTransport(profile, existing_serial=_FakeSerial(_FakeDevice()))
        transport.connect()
    permit_manager = PermitManager()
    arming = ArmingController(permit_manager)
    arming.begin_preflight()
    arming.mark_shadow_validated(**HASHES)
    permit = permit_manager.issue(
        body_id="rh56_left_01",
        **HASHES,
        max_step_delta_raw=max_step_delta_raw,
        max_speed=100,
        max_force_g=100.0,
        expires_in_sec=120.0,
        operator_armed=True,
        physical_estop_confirmed=True,
        execution_mode=mode.value,
    )
    arming.arm(permit.permit_id)
    step = SingleStepExecutor(
        executor=RH56Executor(transport, profile),
        profile=profile,
        permit_manager=permit_manager,
        arming=arming,
        verifier=FeedbackVerifier(profile, calibration),
        execution_mode=mode,
    )
    return step, permit, transport


def _envelope(permit_id: str, values: list[float], **overrides) -> ActionEnvelope:
    args = {
        "permit_id": permit_id,
        "names": ["little", "ring", "middle", "index", "thumb", "thumb_rot"],
        "values": values,
        "representation": "joint_position",
        "units": "raw_device_unit",
        "hashes": HASHES,
        "speed": 100,
        "force_limit_g": 100.0,
        "observation_timestamp_ns": time.monotonic_ns(),
    }
    args.update(overrides.pop("arguments", {}))
    envelope = ActionEnvelope(
        actor_id="pytest",
        agent_framework="pytest",
        session_id="session_test",
        body_id="rh56_left_01",
        capability_id=CAPABILITY_ID,
        arguments=args,
        execution_mode=ExecutionMode.REAL,
        body_snapshot_hash="sha256:snapshot",
        authorization=AuthorizationContext(
            principal_id="operator",
            approved=True,
            approval_id="approval_1",
            scopes=[CAPABILITY_ID],
        ),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.PHYSICALLY_OBSERVED,
            fail_closed=True,
        ),
    )
    for key, value in overrides.items():
        setattr(envelope, key, value)
    return envelope


def test_real_step_through_gateway_completes() -> None:
    step, permit, _ = _armed_stack()
    gateway = ActionGateway()
    gateway.register_executor(CAPABILITY_ID, ExecutionMode.REAL, RH56RealStepExecutor(step))

    receipt = gateway.submit(_envelope(permit.permit_id, [990.0] * 6))

    assert receipt.final_state is ActionState.COMPLETED
    assert receipt.evidence_level is EvidenceLevel.PHYSICALLY_OBSERVED
    assert receipt.driver_ack == {"acknowledged": True}
    assert receipt.verified
    assert receipt.trust_level == "VERIFIED"
    assert receipt.usable_for_real_execution
    assert step.commands_executed == 1
    assert step.hardware_actions_executed == 1
    assert step.fixture_actions_executed == 0


def test_real_step_without_authorization_blocked() -> None:
    step, permit, _ = _armed_stack()
    gateway = ActionGateway()
    gateway.register_executor(CAPABILITY_ID, ExecutionMode.REAL, RH56RealStepExecutor(step))
    envelope = _envelope(permit.permit_id, [990.0] * 6)
    envelope.authorization = AuthorizationContext()

    receipt = gateway.submit(envelope)

    assert receipt.final_state is ActionState.BLOCKED
    assert receipt.errors[0]["code"] == "AUTHORIZATION_REQUIRED"
    assert step.commands_executed == 0


def test_stale_observation_blocks_before_dispatch() -> None:
    step, permit, _ = _armed_stack()
    gateway = ActionGateway()
    gateway.register_executor(CAPABILITY_ID, ExecutionMode.REAL, RH56RealStepExecutor(step))
    envelope = _envelope(
        permit.permit_id,
        [990.0] * 6,
        arguments={"observation_timestamp_ns": time.monotonic_ns() - int(2e9)},
    )

    receipt = gateway.submit(envelope)

    assert receipt.final_state is ActionState.BLOCKED
    assert receipt.errors[0]["code"] == "stale_action"
    assert step.commands_executed == 0


def test_missing_executor_fails_closed() -> None:
    gateway = ActionGateway()
    step, permit, _ = _armed_stack()
    receipt = gateway.submit(_envelope(permit.permit_id, [990.0] * 6))
    assert receipt.final_state is ActionState.FAILED
    assert receipt.errors[0]["code"] == "EXECUTOR_UNAVAILABLE"
    assert step.commands_executed == 0


def test_mode_mismatch_fails_closed_both_directions() -> None:
    # REAL executor wrapping a FIXTURE transport must not dispatch.
    profile = _profile()
    mock_transport = MockModbusTransport(profile)
    mock_transport.connect()
    step, permit, _ = _armed_stack(mode=ExecutionMode.REAL, transport=mock_transport)
    result = step.execute_candidate(
        permit_id=permit.permit_id,
        proposal_id="p1",
        names=list(profile.action_order),
        values=[1000.0] * 6,
        representation="joint_position",
        units="raw_device_unit",
        hashes=HASHES,
        speed=100,
        force_limit_g=100.0,
        observation_timestamp_ns=time.monotonic_ns(),
    )
    assert result.status == "blocked"
    assert result.error_code == "RUNTIME_ACTION_GATEWAY_REQUIRED"

    # FIXTURE executor wrapping a REAL transport must not dispatch either.
    real_transport = SerialModbusTransport(profile, existing_serial=_FakeSerial(_FakeDevice()))
    real_transport.connect()
    step2, permit2, _ = _armed_stack(mode=ExecutionMode.FIXTURE, transport=real_transport)
    result2 = step2.execute_candidate(
        permit_id=permit2.permit_id,
        proposal_id="p2",
        names=list(profile.action_order),
        values=[1000.0] * 6,
        representation="joint_position",
        units="raw_device_unit",
        hashes=HASHES,
        speed=100,
        force_limit_g=100.0,
        observation_timestamp_ns=time.monotonic_ns(),
    )
    assert result2.status == "blocked"
    assert result2.error_code == "RUNTIME_ACTION_GATEWAY_REQUIRED"


def test_step_delta_refusal_faults_and_revokes_permit() -> None:
    step, permit, _ = _armed_stack()
    gateway = ActionGateway()
    gateway.register_executor(CAPABILITY_ID, ExecutionMode.REAL, RH56RealStepExecutor(step))
    # 1000 -> 800 exceeds the permit's max_step_delta_raw=30.
    receipt = gateway.submit(_envelope(permit.permit_id, [800.0] * 6))

    assert receipt.final_state is ActionState.BLOCKED
    assert "step_delta_exceeded" in receipt.errors[0]["code"]
    assert step.commands_executed == 0


def _calibration(tolerance: int = 25):
    from rosclaw.body.rh56.calibration import (
        ActuatorCalibration,
        CalibrationValidation,
        RH56Calibration,
    )

    names = ["little", "ring", "middle", "index", "thumb", "thumb_rot"]
    return RH56Calibration(
        body_id="rh56_left_01",
        transport_profile="test_profile",
        actuators={name: ActuatorCalibration(position_tolerance_raw=tolerance) for name in names},
        validation=CalibrationValidation(status="validated"),
    )


def test_setpoint_hysteresis_keeps_static_joint_setpoints() -> None:
    # thumb_rot sits mid-range (actual == setpoint 900 after the first move).
    # A follow-up command that moves ONLY the index must not re-plan thumb_rot
    # (a setpoint rewrite to ≈current would coast the servo on real hardware).
    from rosclaw.body.rh56 import modbus

    device = _FakeDevice()
    transport = SerialModbusTransport(_profile(), existing_serial=_FakeSerial(device))
    transport.connect()
    step, permit, _ = _armed_stack(
        calibration=_calibration(tolerance=25),
        transport=transport,
        max_step_delta_raw=60.0,
    )

    # First: move thumb_rot to 900 (its setpoint becomes 900, actual snaps to 900).
    transport.write_position([1000] * 5 + [900], speed=100, force_limit=100)
    time.sleep(0.01)
    assert transport.read_state().position[5] == 900

    # Now request index=950 (moving) with thumb_rot=905 (within 25 of actual 900).
    values = [1000.0, 1000.0, 1000.0, 950.0, 1000.0, 905.0]
    result = step.execute_candidate(
        permit_id=permit.permit_id,
        proposal_id="hyst_1",
        names=["little", "ring", "middle", "index", "thumb", "thumb_rot"],
        values=values,
        representation="joint_position",
        units="raw_device_unit",
        hashes=HASHES,
        speed=100,
        force_limit_g=100.0,
        observation_timestamp_ns=time.monotonic_ns(),
    )
    assert result.status == "completed", result.message
    angle_set = device.registers[modbus.Register.ANGLE_SET]
    assert angle_set[3] == 950  # index moved as requested
    assert angle_set[5] == 900  # thumb_rot setpoint preserved (no re-plan)


def test_setpoint_hysteresis_does_not_apply_beyond_tolerance() -> None:
    # thumb_rot target 940 is 40 raw away (beyond tolerance 25): must re-plan.
    from rosclaw.body.rh56 import modbus

    device = _FakeDevice()
    transport = SerialModbusTransport(_profile(), existing_serial=_FakeSerial(device))
    transport.connect()
    step, permit, _ = _armed_stack(
        calibration=_calibration(tolerance=25),
        transport=transport,
        max_step_delta_raw=60.0,
    )
    transport.write_position([1000] * 5 + [900], speed=100, force_limit=100)
    time.sleep(0.01)

    values = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 940.0]
    result = step.execute_candidate(
        permit_id=permit.permit_id,
        proposal_id="hyst_2",
        names=["little", "ring", "middle", "index", "thumb", "thumb_rot"],
        values=values,
        representation="joint_position",
        units="raw_device_unit",
        hashes=HASHES,
        speed=100,
        force_limit_g=100.0,
        observation_timestamp_ns=time.monotonic_ns(),
    )
    assert result.status == "completed", result.message
    assert device.registers[modbus.Register.ANGLE_SET][5] == 940
