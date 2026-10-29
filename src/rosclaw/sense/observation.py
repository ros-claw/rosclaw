"""Pure Sense projection for the daemon; no Runtime creation or ROS transport."""

from typing import Any


class SenseObservationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is not None and hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, dict):
            return result
    return {}


def project_robot_state(sense: Any, robot_id: str) -> dict[str, Any]:
    """Preserve missing, synthetic, stale and historical source semantics.

    Observation readability is not Body binding or action authorization.
    The daemon owns and configures Sense; this function starts no collector.
    """
    if sense is None:
        raise SenseObservationError("SENSE_NOT_CONFIGURED", "Daemon Body Sense is not initialized.")
    try:
        latest = sense.get_latest_state()
        body_sense = sense.get_body_sense()
        readiness = sense.get_readiness()
        body_state = _safe_dict(latest)
        raw_source = body_state.get("source")
        source = raw_source if isinstance(raw_source, str) else ""
        collector = str(getattr(getattr(sense, "config", None), "collector", ""))
        if source.startswith("mock:") or collector == "mock":
            raise SenseObservationError(
                "SYNTHETIC_SOURCE_NOT_ALLOWED",
                "Daemon Sense is synthetic; no live observation is available.",
            )
        if source in {"unconfigured", "file_replay:empty"} or collector == "unconfigured":
            raise SenseObservationError(
                "STATE_PROVENANCE_UNAVAILABLE", f"No configured observation (source={source})."
            )
        if source.startswith("file_replay:"):
            mode, execution_mode, trust_level = "replay", "REPLAY", "RECORDED"
        elif (
            not source.strip()
            or source == "unknown"
            or source.endswith(":stub")
            or source == "sense:degraded"
        ):
            raise SenseObservationError(
                "STATE_PROVENANCE_UNAVAILABLE",
                f"No live provenance (source={source or 'missing'}).",
            )
        else:
            mode = "stale" if getattr(sense, "is_stale", True) else "live"
            execution_mode = "UNKNOWN"
            trust_level = "STALE" if mode == "stale" else "OBSERVED"
        return {
            "robot_id": robot_id,
            "mode": mode,
            "execution_mode": execution_mode,
            "trust_level": trust_level,
            "usable_for_real_execution": False,
            "body_state": body_state,
            "body_sense": _safe_dict(body_sense),
            "readiness": _safe_dict(readiness),
            "is_stale": getattr(sense, "is_stale", True),
            "age_ms": getattr(sense, "state_age_ms", 0),
        }
    except SenseObservationError:
        raise
    except Exception as exc:
        raise SenseObservationError("SENSE_READ_FAILED", str(exc)) from exc
