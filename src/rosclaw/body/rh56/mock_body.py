"""Mock RH56 body construction for shadow/execute validation without hardware."""

from __future__ import annotations

from typing import Any

from rosclaw.body.rh56.transport_profile import TransportProfile


def build_mock_rh56_body(
    profile: TransportProfile,
    *,
    body_instance_id: str = "rh56_mock",
) -> Any:
    """Return an EffectiveBody-like object whose joints mirror the profile.

    Joints carry the canonical action-order names, ``raw_device_unit`` units
    and the profile's position range as limits, so body mapping against the
    RH56 reference policy resolves to ``exact`` compatibility.
    """
    lo, hi = profile.position_range
    joints = {
        name: {
            "type": "revolute",
            "unit": "raw_device_unit",
            "limits": {"lower": float(lo), "upper": float(hi)},
        }
        for name in profile.action_order
    }
    return type(
        "EffectiveBody",
        (),
        {
            "body_instance_id": body_instance_id,
            "joints": joints,
        },
    )()
