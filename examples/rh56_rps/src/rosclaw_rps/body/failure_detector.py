"""BodySense monitoring for the RH56 hand during RPS demo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..gesture_schema import HandTelemetry


@dataclass
class BodySenseVerdict:
    ok: bool
    confidence: float
    failure_type: Optional[str]
    evidence: Dict[str, object]


class FailureDetector:
    """Detect unsafe hand conditions from telemetry."""

    def __init__(
        self,
        max_force_g: float = 500.0,
        max_current_ma: float = 900.0,
        max_temp_c: float = 65.0,
    ):
        self.max_force_g = max_force_g
        self.max_current_ma = max_current_ma
        self.max_temp_c = max_temp_c

    def check(self, telemetry: HandTelemetry) -> BodySenseVerdict:
        evidence: Dict[str, object] = {}
        failure_type = None
        confidence = 1.0

        forces = [abs(v) for v in (telemetry.force_act or {}).values() if v is not None]
        force_peak = max(forces) if forces else 0
        evidence["force_peak"] = force_peak
        if force_peak > self.max_force_g:
            failure_type = "force_over_limit"
            confidence = 0.3

        currents = [v for v in (telemetry.current_ma or {}).values() if v is not None]
        current_peak = max(currents) if currents else 0
        evidence["current_peak"] = current_peak
        if current_peak > self.max_current_ma:
            failure_type = "over_current"
            confidence *= 0.5

        temps = [v for v in (telemetry.temperature_c or {}).values() if v is not None]
        temp_peak = max(temps) if temps else 0
        evidence["temp_peak"] = temp_peak
        if temp_peak > self.max_temp_c:
            failure_type = "temperature_limited"
            confidence *= 0.5

        statuses = [v for v in (telemetry.status or {}).values() if v is not None]
        if any((s & 0x04 or s & 0x08 or s & 0x10) for s in statuses):
            failure_type = "driver_error"
            confidence *= 0.3

        return BodySenseVerdict(ok=failure_type is None, confidence=confidence, failure_type=failure_type, evidence=evidence)
