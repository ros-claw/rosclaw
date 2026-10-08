"""Original owned SIM instrument lift request/response correspondence.

This module never invokes a service. A successful service reply only arms the
probe state machine; measured source transitions remain mandatory.
"""

import hashlib
import math
import re

from backend_probe_evidence import probe_policy


def lift_request(policy):
    """Return bounded protobuf text for the declared instrument alone."""
    probe_policy(policy)
    model = policy["native_policy"]["contact_policy"]["model_name"]
    if type(model) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", model):
        raise ValueError("bounded exact instrument model name required")
    x, y = policy["probe_xy"]
    z = policy["lift_z_m"]
    return (
        f'name: "{model}" position {{ x: {x:.17g} y: {y:.17g} z: {z:.17g} }} '
        "orientation { w: 1 x: 0 y: 0 z: 0 }"
    ).encode("ascii")


def derive_lift_ack(policy, *, request_bytes, response_bytes, returncode, acknowledged_at_unix_ns):
    """Derive an ACK only from retained exact request and successful reply."""
    expected = lift_request(policy)
    if type(request_bytes) is not bytes or request_bytes != expected:
        raise ValueError("original service request differs from frozen instrument target")
    if (
        type(response_bytes) is not bytes
        or not 0 < len(response_bytes) <= 4096
        or not re.fullmatch(rb"\s*data:\s*true\s*", response_bytes)
        or type(returncode) is not int
        or returncode != 0
        or type(acknowledged_at_unix_ns) is not int
        or acknowledged_at_unix_ns <= 0
    ):
        raise ValueError("original successful bounded gz.msgs.Boolean service reply required")
    base = policy["native_policy"]["contact_policy"]
    return {
        "source": "owned_gazebo_world_set_pose_ack_not_pose_proof",
        "run_id": base["run_id"],
        "model_name": base["model_name"],
        "target_xyz": [*policy["probe_xy"], policy["lift_z_m"]],
        "acknowledged_at_unix_ns": acknowledged_at_unix_ns,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "service_success": True,
    }


def lift_service_spec(policy, *, world_name, robot_model_name, wall_timeout_sec):
    """Reviewable argv only; this function does not execute or authorize it.

    Whole-world source ownership, geometry clearance and live ground readiness
    must be independently admitted before a controller may invoke this spec.
    """
    probe_policy(policy)
    for value in (world_name, robot_model_name):
        if type(value) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value):
            raise ValueError("explicit bounded owned world and robot identity required")
    if robot_model_name == policy["native_policy"]["contact_policy"]["model_name"]:
        raise ValueError("SIM instrument service cannot target the active robot")
    if (
        type(wall_timeout_sec) not in (int, float)
        or not math.isfinite(wall_timeout_sec)
        or not 0.1 <= wall_timeout_sec <= 2
    ):
        raise ValueError("bounded immutable scene service timeout required")
    request = lift_request(policy)
    return {
        "evidence_role": "unexecuted_SIM_instrument_scene_service_specification",
        "argv": [
            "gz",
            "service",
            "-s",
            f"/world/{world_name}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(round(wall_timeout_sec * 1000)),
            "--req",
            request.decode("ascii"),
        ],
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "authorization": False,
        "source_ownership_admitted": False,
        "physical_acceptance": "NOT_RUN",
    }
