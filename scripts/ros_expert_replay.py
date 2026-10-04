#!/usr/bin/env python3
"""Produce deterministic offline ROS Expert evidence. No planner or actuation."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from rosclaw.connectors.ros.context import compile_agent_summary
from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.intelligence import build_system_model
from rosclaw.connectors.ros.mission import compile_mission
from rosclaw.connectors.ros.resolver import resolve_task
from rosclaw.connectors.ros.verification.coverage import CleaningPose, CoverageVerifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/ros_expert/nav2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def load(name):
        return json.loads((args.fixtures / f"{name}.json").read_text())

    def write(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    native = load("native")
    now = datetime.fromisoformat(native["captured_at"])
    model = build_system_model(
        RosGraphSnapshot.from_dict(load("graph")),
        robot_id="fixture_base",
        body=load("body"),
        native=native,
    )
    write("system_model.json", model.to_dict())
    write("solution.json", resolve_task(model, "把房间完整清扫一遍", now=now))
    write(
        "mission.json",
        compile_mission(model, "清扫房间", mission_id="fixture_cleaning", now=now).model_dump(
            mode="json"
        ),
    )
    (args.output / "ROS_CONTEXT.md").write_text(compile_agent_summary(model, now=now) + "\n")
    mutations = {
        "tf_missing": lambda m: setattr(m, "transforms", m.transforms[1:]),
        "sensor_stale": lambda m: setattr(m.signals[0], "last_message_age_ms", 2000),
        "qos_fault": lambda m: m.qos.update(
            endpoints=[
                {"topic": "/scan", "kind": "publisher", "reliability": "BEST_EFFORT"},
                {"topic": "/scan", "kind": "subscriber", "reliability": "RELIABLE"},
            ]
        ),
        "lifecycle_fault": lambda m: setattr(m.lifecycle[0], "state", "INACTIVE"),
        "costmap_fault": lambda m: m.navigation.update(obstacle_source_configured=False),
        "sim_time_fault": lambda m: m.observations["node_use_sim_time"].update({"/amcl": False}),
    }
    diagnoses = {"normal": diagnose(model, now=now)}
    for name, mutate in mutations.items():
        fault = model.model_copy(deep=True)
        mutate(fault)
        fault.seal()
        diagnoses[name] = diagnose(fault, now=now)
    write("diagnostic_results.json", {"evidence_domain": "fixture_replay", "scenarios": diagnoses})
    verifier = CoverageVerifier(
        width=10,
        height=10,
        resolution=0.1,
        accessible_cells=list(range(100)),
        cleaning_polygon=[(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
    )
    verifier.set_temporary_blocked(list(range(40, 60)))
    trajectory, stamp = [], 0.0
    for row in range(10):
        for col in range(10) if row % 2 == 0 else reversed(range(10)):
            pose = CleaningPose((col + 0.5) * 0.1, (row + 0.5) * 0.1, 0, stamp, True)
            verifier.observe(pose, frame_id="map")
            trajectory.append(pose.__dict__)
            stamp += 0.1
    blocked = verifier.result()
    verifier.set_temporary_blocked([])
    for row in [4, 5]:
        for col in range(10):
            stamp += 0.1
            pose = CleaningPose((col + 0.5) * 0.1, (row + 0.5) * 0.1, 0, stamp, True)
            verifier.observe(pose, frame_id="map")
            trajectory.append(pose.__dict__)
    repaired = verifier.result()
    write("trajectory.json", {"evidence_domain": "synthetic_trace", "poses": trajectory})
    write(
        "coverage_results.json",
        {
            "evidence_domain": "synthetic_trace",
            "blocked": blocked,
            "repaired": repaired,
            "collision_count": None,
            "mission_verification": "NOT_VERIFIED",
            "nav2_acceptance": "NOT_RUN",
            "note": "Trace rasterization exercise; no coverage planner or robot executed.",
        },
    )
    print(f"Offline replay evidence written to {args.output}")


if __name__ == "__main__":
    main()
