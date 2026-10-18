"""Pure SDK-owned new-Body policy selection, never a Permit or motion entry."""

import copy
import json

from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding


def select_navigation_fixture_policy(policy, operator_input, *, action_name, frame_id):
    """Preserve legacy defaults or bind explicit limits to this original source.

    Called only by the existing owned isolated SDK fixture before immutable Body
    compilation. No existing Body is edited; the existing compiler still checks
    all policy inputs and independently binds the new snapshot. This is not an
    Agent interface, live policy update, source authentication or action approval.
    """
    baseline = copy.deepcopy(policy)
    if operator_input is None:
        return baseline
    if type(operator_input) is not dict or set(operator_input) != {
        "schema_version",
        "scope",
        "policy_id",
        "body_id",
        "source_urdf_sha256",
        "navigation_requirements_hash",
        "navigation_limits",
    }:
        raise ValueError("exact bounded SDK operator policy input required")
    if len(json.dumps(operator_input, allow_nan=False).encode()) > 8192:
        raise ValueError("bounded operator policy input required")
    if (
        operator_input["schema_version"] != "rosclaw.sdk_navigation_policy_input.v1"
        or operator_input["scope"] != "OPERATOR_OWNED_DISPOSABLE_SIM_NEW_BODY"
    ):
        raise ValueError("explicit owned disposable SIM new-Body policy scope required")
    for key in ("body_id", "source_urdf_sha256", "navigation_requirements_hash"):
        if operator_input[key] != baseline[key]:
            raise ValueError(
                "operator policy differs from current original Body/source binding: " + key
            )
    identity = operator_input["policy_id"]
    if (
        type(identity) is not str
        or not 1 <= len(identity) <= 128
        or any(ord(c) < 32 for c in identity)
    ):
        raise ValueError("bounded explicit operator policy identity required")
    limits = operator_input["navigation_limits"]
    if type(limits) is not dict or set(limits) != set(baseline["navigation_limits"]):
        raise ValueError("all and only the existing eight explicit navigation limits required")
    # The same pure constructor used by the existing Body compiler checks finite
    # positive numbers, bool rejection, endpoint identities and stop thresholds.
    # The zero hash is a validation placeholder, never a compiled Body claim.
    witness = baseline["witness_declaration"]
    SimulationNavigationBinding(
        baseline["body_id"],
        "0" * 64,
        action_name,
        witness["lease_service"],
        witness["observation_topic"],
        frame_id,
        witness["run_id"],
        **limits,
    )
    baseline["policy_id"] = identity
    baseline["navigation_limits"] = copy.deepcopy(limits)
    return json.loads(json.dumps(baseline, allow_nan=False))
