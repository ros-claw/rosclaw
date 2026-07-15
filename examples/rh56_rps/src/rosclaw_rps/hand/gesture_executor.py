"""Execute and verify RH56 gestures."""
from __future__ import annotations

import threading
import time
from typing import Dict, List

from .rh56_controller import HandController
from ..gesture_schema import (
    GestureConfig,
    GestureExecutionResult,
    HandTelemetry,
    VerificationResult,
)


class GestureVerifier:
    """Heuristic verification that the hand actually achieved the target gesture."""

    def verify(self, config: GestureConfig, telemetry: HandTelemetry) -> VerificationResult:
        evidence: Dict[str, object] = {}
        failure_type = None
        confidence = 1.0

        target = {name: config.angles[i] for i, name in enumerate(self._dof_names())}
        actual = telemetry.angle_actual or {}
        angle_errors = []
        for name, target_val in target.items():
            actual_val = actual.get(name)
            if actual_val is None:
                angle_errors.append(10000)
                continue
            angle_errors.append(abs(actual_val - target_val))
        max_error = max(angle_errors) if angle_errors else 10000
        evidence["joint_error_max"] = max_error
        evidence["joint_errors"] = {n: e for n, e in zip(target.keys(), angle_errors)}

        if max_error > config.verify_tolerance:
            failure_type = "joint_not_reached"
            confidence = max(0.0, 1.0 - (max_error - config.verify_tolerance) / 500.0)

        currents = telemetry.current_ma or {}
        current_values = [v for v in currents.values() if v is not None]
        current_peak = max(current_values) if current_values else 0
        evidence["current_peak"] = current_peak
        if current_peak > config.verify_current_max:
            failure_type = "over_current"
            confidence *= 0.5

        forces = telemetry.force_act or {}
        force_values = [abs(v) for v in forces.values() if v is not None]
        force_peak = max(force_values) if force_values else 0
        evidence["force_peak"] = force_peak
        if force_peak > config.verify_force_max:
            failure_type = "force_over_limit"
            confidence *= 0.5

        statuses = telemetry.status or {}
        status_values = [v for v in statuses.values() if v is not None]
        if config.verify_driver_status and any(
            (s & 0x04 or s & 0x08 or s & 0x10) for s in status_values
        ):
            failure_type = "driver_error"
            confidence *= 0.3

        if failure_type is None:
            confidence = min(1.0, 1.0 - max_error / (2.0 * config.verify_tolerance))

        ok = failure_type is None
        return VerificationResult(ok=ok, confidence=round(confidence, 3), failure_type=failure_type, evidence=evidence)

    @staticmethod
    def _dof_names() -> List[str]:
        return ["little", "ring", "middle", "index", "thumb", "thumb_rot"]


class GestureExecutor:
    """High-level gesture executor with verification and safe recovery."""

    def __init__(self, hand: HandController, gestures: Dict[str, GestureConfig], verifier: GestureVerifier):
        self.hand = hand
        self.gestures = gestures
        self.verifier = verifier
        self._lock = threading.Lock()

    def execute(self, name: str) -> GestureExecutionResult:
        config = self.gestures.get(name)
        if config is None:
            return GestureExecutionResult(
                gesture_name=name,
                command_success=False,
                verified=False,
                failure_reason=f"unknown_gesture:{name}",
            )

        with self._lock:
            start = time.time()
            ok = self.hand.move_to_gesture(name, config.angles, config.speed, config.force)
            if not ok:
                self.hand.safe_open()
                return GestureExecutionResult(
                    gesture_name=name,
                    command_success=False,
                    verified=False,
                    failure_reason="move_command_failed",
                    start_time=start,
                    end_time=time.time(),
                )

            time.sleep(config.hold_s)
            telemetry = self.hand.read_telemetry()
            verdict = self.verifier.verify(config, telemetry)

            if not verdict.ok:
                self.hand.safe_open()

            return GestureExecutionResult(
                gesture_name=name,
                command_success=True,
                verified=verdict.ok,
                telemetry=telemetry,
                failure_reason=verdict.failure_type,
                start_time=start,
                end_time=time.time(),
            )
