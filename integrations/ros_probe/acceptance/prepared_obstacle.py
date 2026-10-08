"""Fixture-only movement of a preloaded obstacle; no robot control or credit.

Run only inside an owned disposable dynamic fixture. Service acknowledgement
is a perturbation log, never proof of actual pose or successful cleaning.
"""

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

from faults import observed, service


class PlacementClearanceUnavailableError(ValueError):
    """Fresh source is valid, but this scene-only move must wait for clearance."""


def pose_request(fixture, body, sample, *, name, x, y):
    """Keep the closed scene and collision IDs intact; screen actual body clearance."""
    if (
        fixture.get("schema_version") != "rosclaw.sim_physics_fixture.v1"
        or type(name) is not str
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
        or fixture["binding"]["body_snapshot_hash"] != body["effective_body_hash"]
        or name not in fixture["binding"]["obstacle_names"]
    ):
        raise ValueError("prepared obstacle and compiled Body binding required")
    matches = [o for o in fixture["obstacles"] if o["name"] == name]
    if len(matches) != 1:
        raise ValueError("exactly one prepared obstacle required")
    obstacle = matches[0]
    values = [
        x,
        y,
        body["physical_radius_m"],
        sample["x"],
        sample["y"],
        *obstacle["box_size"],
        *obstacle["pose"],
    ]
    if any(type(v) not in (int, float) or not -100 <= v <= 100 for v in values):
        raise ValueError("bounded finite fixture geometry and pose required")
    if (
        sample["observation_complete"] is not True
        or type(sample["collision_count"]) is not int
        or sample["collision_count"] != 0
    ):
        raise ValueError("complete contact-free independent observation required")
    captured = datetime.fromisoformat(sample["captured_at"])
    if not 0 <= (datetime.now(UTC) - captured).total_seconds() < 0.3:
        raise ValueError("fresh independent body pose required")
    radius = body["physical_radius_m"]
    size = obstacle["box_size"]
    if not 0 < radius <= 10 or any(v <= 0 for v in size):
        raise ValueError("positive actual fixture dimensions required")
    bound = math.hypot(*size) / 2
    if math.hypot(x - sample["x"], y - sample["y"]) <= radius + bound + 0.1:
        raise PlacementClearanceUnavailableError("obstacle movement would violate actual body clearance")
    z, roll, pitch, yaw = obstacle["pose"][2:]
    if any(v != 0 for v in (roll, pitch, yaw)):
        raise ValueError("this fixture movement supports frozen axis-aligned boxes only")
    return (
        f'name: "{name}" position {{ x: {x:.17g} y: {y:.17g} z: {z:.17g} }} orientation {{ w: 1 }}'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--x", required=True, type=float)
    parser.add_argument("--y", required=True, type=float)
    parser.add_argument("--record", required=True, type=Path)
    args = parser.parse_args()
    root = Path("/evidence")
    fixture = json.loads((root / "physics_fixture.json").read_text())
    body = json.loads((root / "body.json").read_text())
    ready = json.loads((root / "physics_ready.json").read_text())
    if ready["binding"] != fixture["binding"]:
        raise ValueError("actual passive source must admit the same closed scene")
    sample = observed()
    request = pose_request(fixture, body, sample, name=args.name, x=args.x, y=args.y)
    # Reserve the record before mutation; never silently overwrite a prior attempt.
    with args.record.open("x") as output:
        record = {
            "evidence_role": "fixture_perturbation_not_physical_acceptance",
            "before": sample,
            "request": request,
            "captured_at": datetime.now(UTC).isoformat(),
            "physical_acceptance": "NOT_RUN",
        }
        output.write(json.dumps(record) + "\n")
        output.flush()
        response = service("set_pose", "gz.msgs.Pose", request)
        output.write(
            json.dumps(
                {
                    "service_response": response,
                    "actual_pose_confirmation": "REQUIRES_INDEPENDENT_POSTUPDATE_PACKET",
                }
            )
            + "\n"
        )
    print(json.dumps({"acknowledged": True, "physical_acceptance": "NOT_RUN"}))


if __name__ == "__main__":
    main()
