"""RH56 calibration schema ``rosclaw.rh56.calibration.v1`` and the calibration gate.

P5 requires a validated calibration before any execution permit can be issued.
The calibration binds to a specific transport profile and records per-actuator
safe ranges, direction, tolerances, plus feedback thresholds (force baseline,
temperature warning/stop).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rosclaw.body.rh56.transport_profile import TransportProfile

RH56_CALIBRATION_SCHEMA_VERSION = "rosclaw.rh56.calibration.v1"

CALIBRATION_STATUSES = ("uncalibrated", "provisional", "validated", "expired")


class CalibrationError(ValueError):
    """Raised when calibration content or status is not acceptable.

    Messages start with a machine-readable code, e.g. ``calibration_not_validated``.
    """


@dataclass
class ActuatorCalibration:
    """Per-actuator calibration record (raw device units)."""

    open_raw: int = 1000
    closed_raw: int = 0
    safe_min_raw: int = 100
    safe_max_raw: int = 1000
    direction: int = -1
    position_tolerance_raw: int = 25


@dataclass
class FeedbackCalibration:
    """Feedback thresholds for force / temperature."""

    force_baseline_file: str = ""
    force_soft_limit_g: float = 100.0
    force_hard_limit_g: float = 300.0
    temperature_warning_c: float = 55.0
    temperature_stop_c: float = 60.0
    thresholds_source: str = "conservative_default"


@dataclass
class CalibrationValidation:
    """Validation block of the calibration document."""

    status: str = "uncalibrated"
    validated_at: str = ""
    rounds: int = 0
    body_hash: str = ""
    transport_profile_hash: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class RH56Calibration:
    """A ``rosclaw.rh56.calibration.v1`` document."""

    body_id: str
    transport_profile: str
    actuators: dict[str, ActuatorCalibration] = field(default_factory=dict)
    feedback: FeedbackCalibration = field(default_factory=FeedbackCalibration)
    validation: CalibrationValidation = field(default_factory=CalibrationValidation)
    schema_version: str = RH56_CALIBRATION_SCHEMA_VERSION

    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RH56Calibration":
        schema = data.get("schema_version")
        if schema != RH56_CALIBRATION_SCHEMA_VERSION:
            raise CalibrationError(
                f"calibration_schema_mismatch: {schema!r} != {RH56_CALIBRATION_SCHEMA_VERSION!r}"
            )
        actuators = {
            name: ActuatorCalibration(**(spec or {}))
            for name, spec in (data.get("actuators") or {}).items()
        }
        feedback = FeedbackCalibration(**(data.get("feedback") or {}))
        validation = CalibrationValidation(**(data.get("validation") or {}))
        calib = cls(
            body_id=str(data.get("body_id", "")),
            transport_profile=str(data.get("transport_profile", "")),
            actuators=actuators,
            feedback=feedback,
            validation=validation,
            schema_version=schema,
        )
        return calib

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    @property
    def status(self) -> str:
        return self.validation.status

    def actuator_names(self) -> list[str]:
        return list(self.actuators.keys())

    # ------------------------------------------------------------------

    def validate_against_profile(self, profile: TransportProfile) -> None:
        """Fail-closed consistency check between calibration and transport profile."""
        if self.transport_profile != profile.id:
            raise CalibrationError(
                f"calibration_transport_mismatch: calibration references "
                f"{self.transport_profile!r} but profile is {profile.id!r}"
            )
        missing = [n for n in profile.action_order if n not in self.actuators]
        if missing:
            raise CalibrationError(
                f"calibration_incomplete: actuators missing calibration: {missing}"
            )
        lo, hi = profile.position_range
        for name in profile.action_order:
            spec = self.actuators[name]
            for field_name in ("open_raw", "closed_raw", "safe_min_raw", "safe_max_raw"):
                value = int(getattr(spec, field_name))
                if not (lo <= value <= hi):
                    raise CalibrationError(
                        f"calibration_range_violation: {name}.{field_name}={value} outside "
                        f"transport range [{lo}, {hi}]"
                    )
            if not (spec.safe_min_raw <= spec.safe_max_raw):
                raise CalibrationError(
                    f"calibration_range_violation: {name} safe range inverted "
                    f"({spec.safe_min_raw} > {spec.safe_max_raw})"
                )

    def clamp_to_safe(self, name: str, value: float) -> int:
        """Clamp a raw target position into the actuator's calibrated safe range."""
        spec = self.actuators.get(name)
        if spec is None:
            raise CalibrationError(f"calibration_incomplete: no calibration for {name!r}")
        return max(spec.safe_min_raw, min(spec.safe_max_raw, int(round(value))))

    def position_tolerance(self, name: str) -> int:
        spec = self.actuators.get(name)
        return spec.position_tolerance_raw if spec is not None else 25


def calibration_has_mock_evidence(calib: RH56Calibration) -> bool:
    """True when the validation evidence came from the mock transport.

    A calibration validated against the mock device must never arm real
    hardware; arming paths check this and refuse unless mock mode is explicit.
    """
    return any("mock=true" in str(item).lower() for item in calib.validation.evidence)


def load_rh56_calibration(path: str | Path) -> RH56Calibration:
    """Load a calibration document from YAML or JSON, fail-closed on errors."""
    path = Path(path)
    if not path.exists():
        raise CalibrationError(f"calibration_not_found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise CalibrationError(f"calibration_schema_mismatch: {path} is not a mapping")
    return RH56Calibration.from_dict(data)


def write_rh56_calibration(calib: RH56Calibration, path: str | Path) -> Path:
    """Write a calibration document to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(calib.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Calibration gate
# ---------------------------------------------------------------------------


class RH56CalibrationGate:
    """Gate that decides whether calibration allows an execution permit."""

    def __init__(self, calibration: RH56Calibration, profile: TransportProfile):
        self.calibration = calibration
        self.profile = profile

    def check(self) -> None:
        """Raise :class:`CalibrationError` unless calibration is validated and consistent."""
        self.calibration.validate_against_profile(self.profile)
        status = self.calibration.status
        if status != "validated":
            raise CalibrationError(
                f"calibration_not_validated: status {status!r}; execution permit denied"
            )

    def mark_validated(
        self,
        *,
        rounds: int,
        body_hash: str = "",
        evidence: list[str] | None = None,
    ) -> RH56Calibration:
        """Return a copy of the calibration marked as validated (in-memory only)."""
        calib = RH56Calibration.from_dict(self.calibration.to_dict())
        calib.validate_against_profile(self.profile)
        if rounds <= 0:
            raise CalibrationError("calibration_not_validated: rounds must be positive")
        calib.validation = CalibrationValidation(
            status="validated",
            validated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            rounds=int(rounds),
            body_hash=body_hash,
            transport_profile_hash=self.profile.content_hash(),
            evidence=list(evidence or []),
        )
        return calib
