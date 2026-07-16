"""Unit conversion utilities for body action mapping.

Only a small, conservative set of conversions is supported.  Anything not in
``SUPPORTED_CONVERSIONS`` is treated as incompatible so the mapper fails closed.
"""

from __future__ import annotations

from typing import Callable

UnitConverter = Callable[[float], float]

# Normalized unit tokens.
RADIAN_ALIASES = {"rad", "radians", "radian"}
DEGREE_ALIASES = {"deg", "degrees", "degree"}
METER_ALIASES = {"m", "meters", "meter", "metres", "metre"}
MILLIMETER_ALIASES = {"mm", "millimeters", "millimeter"}


def _normalize(unit: str) -> str:
    token = unit.strip().lower()
    if token in RADIAN_ALIASES:
        return "rad"
    if token in DEGREE_ALIASES:
        return "deg"
    if token in METER_ALIASES:
        return "m"
    if token in MILLIMETER_ALIASES:
        return "mm"
    return token


def _identity(value: float) -> float:
    return value


def _rad_to_deg(value: float) -> float:
    import math

    return value * 180.0 / math.pi


def _deg_to_rad(value: float) -> float:
    import math

    return value * math.pi / 180.0


def _m_to_mm(value: float) -> float:
    return value * 1000.0


def _mm_to_m(value: float) -> float:
    return value / 1000.0


SUPPORTED_CONVERSIONS: dict[tuple[str, str], UnitConverter] = {
    ("rad", "rad"): _identity,
    ("deg", "deg"): _identity,
    ("m", "m"): _identity,
    ("mm", "mm"): _identity,
    ("rad", "deg"): _rad_to_deg,
    ("deg", "rad"): _deg_to_rad,
    ("m", "mm"): _m_to_mm,
    ("mm", "m"): _mm_to_m,
}


def get_unit_conversion(from_unit: str, to_unit: str) -> UnitConverter | None:
    """Return a converter if the unit transition is supported, else None."""
    from_norm = _normalize(from_unit)
    to_norm = _normalize(to_unit)
    if from_norm == to_norm:
        return _identity
    return SUPPORTED_CONVERSIONS.get((from_norm, to_norm))


def units_compatible(from_unit: str, to_unit: str) -> bool:
    """Return True if ``from_unit`` can be converted to ``to_unit``."""
    return get_unit_conversion(from_unit, to_unit) is not None
