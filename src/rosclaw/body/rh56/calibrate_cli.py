"""CLI handlers for ``rosclaw body calibrate-rh56`` and ``validate-calibration``.

Calibration uses a transport to probe the hand.  Until the physical device is
available the ``--mock`` flag drives :class:`MockModbusTransport`; without
``--mock`` the real :class:`SerialModbusTransport` is constructed, which is
fail-closed when the device path is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw.body.rh56.calibration import (
    ActuatorCalibration,
    CalibrationError,
    FeedbackCalibration,
    RH56Calibration,
    RH56CalibrationGate,
    load_rh56_calibration,
    write_rh56_calibration,
)
from rosclaw.body.rh56.transport import (
    MockModbusTransport,
    RH56Transport,
    SerialModbusTransport,
    TransportUnavailableError,
)
from rosclaw.body.rh56.transport_profile import (
    TransportBindingError,
    TransportProfile,
    load_transport_profile,
    validate_transport_binding,
)


def _build_transport(profile: TransportProfile, mock: bool) -> RH56Transport:
    if mock:
        mock_transport = MockModbusTransport(profile)
        mock_transport.connect()
        return mock_transport
    try:
        serial_transport = SerialModbusTransport(profile)
        serial_transport.connect()
        return serial_transport
    except TransportUnavailableError as exc:
        raise CalibrationError(f"transport_unavailable: {exc}") from exc


def cmd_body_calibrate_rh56(args: argparse.Namespace) -> int:
    """Generate an RH56 calibration document for a body."""
    try:
        profile = load_transport_profile(args.transport_profile)
    except TransportBindingError as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    if not args.body or not args.body.strip():
        print("[ROSClaw] --body is required")
        return 1

    actuators = {}
    probe_positions: list[int] | None = None
    if not args.no_probe:
        try:
            transport = _build_transport(profile, mock=args.mock)
            feedback = transport.read_state()
            probe_positions = feedback.position
            transport.close()
        except CalibrationError as exc:
            print(f"[ROSClaw] probe failed: {exc}")
            return 1

    for i, name in enumerate(profile.action_order):
        spec = ActuatorCalibration()
        if probe_positions is not None:
            # Record the probed resting position as evidence; safe range stays
            # conservative until an operator reviews it.
            spec.open_raw = max(probe_positions[i], spec.open_raw)
        actuators[name] = spec

    calib = RH56Calibration(
        body_id=args.body,
        transport_profile=profile.id,
        actuators=actuators,
        feedback=FeedbackCalibration(),
    )

    out = Path(args.output).expanduser()
    write_rh56_calibration(calib, out)
    print(f"[ROSClaw] RH56 calibration written: {out}")
    print(f"  body: {args.body}")
    print(f"  transport profile: {profile.id}")
    print(f"  actuators: {len(actuators)}")
    print("  status: uncalibrated (run `rosclaw body validate-calibration` with the hand)")
    return 0


def cmd_body_validate_calibration(args: argparse.Namespace) -> int:
    """Validate an RH56 calibration against its transport profile."""
    try:
        profile = load_transport_profile(args.transport_profile)
        calib = load_rh56_calibration(args.calibration)
    except (TransportBindingError, CalibrationError) as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    if calib.body_id != args.body:
        print(
            f"[ROSClaw] calibration body_id {calib.body_id!r} does not match --body {args.body!r}"
        )
        return 1

    try:
        validate_transport_binding(
            profile,
            provider_ref=args.provider_ref or profile.metadata.get("provider_ref"),
            device_path=args.device or None,
        )
    except TransportBindingError as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    gate = RH56CalibrationGate(calib, profile)
    try:
        calib.validate_against_profile(profile)
    except CalibrationError as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    # Probe rounds: every round must succeed with stable dimensions.
    rounds = int(args.rounds)
    try:
        transport = _build_transport(profile, mock=args.mock)
    except CalibrationError as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    read_ok = 0
    try:
        for _ in range(rounds):
            feedback = transport.read_state()
            if len(feedback.position) == profile.command.actuator_count and feedback.ok:
                read_ok += 1
    finally:
        transport.close()

    success = read_ok == rounds
    result = {
        "body_id": args.body,
        "transport_profile": profile.id,
        "rounds": rounds,
        "read_ok": read_ok,
        "success": success,
        "mock": bool(args.mock),
    }

    if success:
        validated = gate.mark_validated(
            rounds=rounds,
            body_hash=args.body_hash or "",
            evidence=[f"probe_rounds={read_ok}/{rounds}", f"mock={bool(args.mock)}"],
        )
        write_rh56_calibration(validated, args.calibration)
        result["status"] = "validated"
    else:
        result["status"] = "failed"

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[ROSClaw] calibration validation: {result['status']}")
        print(f"  rounds: {read_ok}/{rounds}")
        print(f"  transport profile: {profile.id}")

    if args.strict and not success:
        print("[ROSClaw] strict mode: calibration NOT validated; execution permit denied")
        return 1
    return 0


def add_rh56_calibration_parsers(body_subparsers) -> None:
    """Register RH56 calibration subcommands on the body parser."""
    cal_rh56 = body_subparsers.add_parser(
        "calibrate-rh56", help="Generate an RH56 calibration document"
    )
    cal_rh56.add_argument("--body", required=True, help="Body instance ID (e.g. rh56_right_01)")
    cal_rh56.add_argument(
        "--transport-profile", required=True, help="Path to transport profile YAML"
    )
    cal_rh56.add_argument("--output", required=True, help="Output calibration YAML path")
    cal_rh56.add_argument(
        "--mock", action="store_true", help="Probe via the mock transport (no hardware)"
    )
    cal_rh56.add_argument(
        "--no-probe", action="store_true", help="Do not probe the device; skeleton only"
    )

    val_cal = body_subparsers.add_parser(
        "validate-calibration", help="Validate an RH56 calibration (rounds of reads)"
    )
    val_cal.add_argument("--body", required=True, help="Body instance ID")
    val_cal.add_argument("--calibration", required=True, help="Path to calibration YAML")
    val_cal.add_argument(
        "--transport-profile", required=True, help="Path to transport profile YAML"
    )
    val_cal.add_argument("--rounds", type=int, default=5, help="Probe rounds (default 5)")
    val_cal.add_argument("--strict", action="store_true", help="Fail unless validated")
    val_cal.add_argument("--mock", action="store_true", help="Probe via the mock transport")
    val_cal.add_argument("--device", default=None, help="Expected device path check")
    val_cal.add_argument("--provider-ref", default=None, help="Expected provider_ref check")
    val_cal.add_argument("--body-hash", default="", help="Body content hash to record")
    val_cal.add_argument("--json", action="store_true", help="JSON output")
