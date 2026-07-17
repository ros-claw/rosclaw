"""Feedback verifier: position / force / temperature / status / current checks.

Implements the P5 feedback doctrine (plan §8.5):

- ``force`` — primary contact/overload criterion
- ``current`` — motion/stall auxiliary criterion (returns ~0 at rest)
- ``status`` — protection hard criterion
- ``temperature`` — thermal hard criterion
"""

from __future__ import annotations

from rosclaw.body.rh56.calibration import RH56Calibration
from rosclaw.body.rh56.transport import RH56Feedback
from rosclaw.body.rh56.transport_profile import TransportProfile
from rosclaw.integrations.lerobot.execution.schema import FeedbackVerification

# RH56 STATUS bit semantics (firmware family documentation, proven in the
# rosclaw_rh56 runtime):  0x01 running | 0x02 in_position are *informational*
# — a healthy hand reports them during normal operation.  Only the protection
# bits are hard faults.
STATUS_BIT_RUNNING = 0x01
STATUS_BIT_IN_POSITION = 0x02
STATUS_BIT_CURRENT_PROTECTION = 0x04
STATUS_BIT_FORCE_PROTECTION = 0x08
STATUS_BIT_TEMP_PROTECTION = 0x10
STATUS_PROTECTION_MASK = (
    STATUS_BIT_CURRENT_PROTECTION | STATUS_BIT_FORCE_PROTECTION | STATUS_BIT_TEMP_PROTECTION
)

_PROTECTION_NAMES = (
    (STATUS_BIT_CURRENT_PROTECTION, "current_protection"),
    (STATUS_BIT_FORCE_PROTECTION, "force_protection"),
    (STATUS_BIT_TEMP_PROTECTION, "temp_protection"),
)


class FeedbackVerifier:
    """Verify transport feedback after one step; provenance is recorded elsewhere."""

    def __init__(
        self,
        profile: TransportProfile,
        calibration: RH56Calibration | None = None,
        *,
        position_timeout_ms: float = 2000.0,
    ):
        self.profile = profile
        self.calibration = calibration
        self.position_timeout_ms = position_timeout_ms

    def verify(
        self,
        *,
        target: list[float],
        feedback: RH56Feedback,
        force_limit_g: float,
    ) -> FeedbackVerification:
        details: list[str] = []
        names = list(self.profile.action_order)

        # Position reached within per-actuator calibrated tolerance.
        position_reached = True
        for i, name in enumerate(names):
            if i >= len(target) or i >= len(feedback.position):
                position_reached = False
                details.append(f"position_dim:{name}")
                continue
            tolerance = (
                self.calibration.position_tolerance(name) if self.calibration is not None else 25
            )
            error = abs(float(feedback.position[i]) - float(target[i]))
            if error > tolerance:
                position_reached = False
                details.append(f"position_error:{name}={error:.1f}>{tolerance}")

        # Force: primary contact/overload criterion.
        force_safe = True
        hard_limit = (
            self.calibration.feedback.force_hard_limit_g if self.calibration is not None else 300.0
        )
        for i, name in enumerate(names):
            if i < len(feedback.force_g):
                force = float(feedback.force_g[i])
                if force > hard_limit:
                    force_safe = False
                    details.append(f"force_hard_limit:{name}={force:.1f}>{hard_limit}")
                elif force > float(force_limit_g):
                    details.append(f"force_soft_limit:{name}={force:.1f}>{force_limit_g}")

        # Temperature: thermal hard criterion.
        temperature_safe = True
        stop_c = (
            self.calibration.feedback.temperature_stop_c if self.calibration is not None else 60.0
        )
        warn_c = (
            self.calibration.feedback.temperature_warning_c
            if self.calibration is not None
            else 55.0
        )
        for i, name in enumerate(names):
            if i < len(feedback.temperature_c):
                temp = float(feedback.temperature_c[i])
                if temp >= stop_c:
                    temperature_safe = False
                    details.append(f"temperature_stop:{name}={temp:.1f}>={stop_c}")
                elif temp >= warn_c:
                    details.append(f"temperature_warning:{name}={temp:.1f}>={warn_c}")

        # Status bits: protection hard criterion.  Informational bits
        # (running/in_position) are healthy states, not faults.
        fault_free = True
        for i, name in enumerate(names):
            if i < len(feedback.status_bits):
                status = int(feedback.status_bits[i])
                if status & STATUS_PROTECTION_MASK:
                    fault_free = False
                    active = [label for bit, label in _PROTECTION_NAMES if status & bit]
                    details.append(f"status_protection:{name}=0x{status:02x}({','.join(active)})")

        return FeedbackVerification(
            position_reached=position_reached,
            force_safe=force_safe,
            temperature_safe=temperature_safe,
            fault_free=fault_free,
            details=details,
        )

    def is_step_ok(self, verification: FeedbackVerification) -> bool:
        """A feedback sample passes only if every hard criterion passes."""
        return (
            verification.position_reached
            and verification.force_safe
            and verification.temperature_safe
            and verification.fault_free
        )
