"""Compile two distinct installed vendor models through the existing Body path.

This is an offline contract check, not cross-Body physical mission acceptance.
The conservative radii and simulated cleaners are explicit fixture declarations.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from fixture_body import configure_fixture_body


def run(waffle: Path, burger: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for path, model, radius, cleaner in (
        (waffle, "turtlebot3_waffle", 0.25, 0.275),
        (burger, "turtlebot3_burger", 0.15, 0.175),
    ):
        robot = ET.fromstring(path.read_bytes())
        if robot.get("name") != model:
            raise ValueError("the actual distinct vendor model is required")
        box = robot.find("link[@name='base_link']/collision/geometry/box").get("size")
        left = robot.find("joint[@name='wheel_left_joint']/origin").get("xyz")
        right = robot.find("joint[@name='wheel_right_joint']/origin").get("xyz")
        separation = abs(float(left.split()[1]) - float(right.split()[1]))
        specification = {
            "body_id": model + "_contract",
            "physical_radius_m": radius,
            "cleaning_polygon": [
                [-cleaner, -cleaner],
                [cleaner, -cleaner],
                [cleaner, cleaner],
                [-cleaner, cleaner],
            ],
        }
        home = output / model
        body_hash = configure_fixture_body(home, specification, path)
        records.append(
            {
                "model": model,
                "urdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "body_hash": body_hash,
                "vendor_base_collision_box_m": box,
                "vendor_wheel_separation_m": separation,
                "declared_conservative_radius_m": radius,
                "simulated_cleaner_half_width_m": cleaner,
            }
        )
    if (
        records[0]["body_hash"] == records[1]["body_hash"]
        or records[0]["urdf_sha256"] == records[1]["urdf_sha256"]
        or records[0]["vendor_base_collision_box_m"] == records[1]["vendor_base_collision_box_m"]
        or records[0]["vendor_wheel_separation_m"] == records[1]["vendor_wheel_separation_m"]
    ):
        raise RuntimeError("the second Body must differ in physical vendor geometry")
    result = {
        "status": "PASS",
        "evidence_domain": "OFFLINE_BODY_CONTRACT",
        "physical_cross_body_mission_verified": False,
        "automatic_unknown_body_integration_verified": False,
        "models": records,
    }
    (output / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("waffle", "burger", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.waffle, args.burger, args.output)
