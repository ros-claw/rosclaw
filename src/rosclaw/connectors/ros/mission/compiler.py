"""Compile semantic coverage requirements into the existing TaskGraphV1."""

from datetime import datetime

from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.connectors.ros.resolver import resolve_task
from rosclaw.contracts.agent.task_graph import (
    TaskConstraints,
    TaskGraphV1,
    TaskKind,
    TaskNodeV1,
    TaskVerification,
)


def compile_mission(
    model: RosSystemModel, task: str, *, mission_id: str, now: datetime | None = None
) -> TaskGraphV1:
    solution = resolve_task(model, task, now=now)
    if solution["task_class"] != "complete_area_cleaning":
        raise ValueError("this mission compiler supports complete area cleaning")
    stages = [
        (
            "precheck",
            TaskKind.VALIDATE,
            [],
            "Inspect readiness, body footprint, map and fresh evidence.",
        ),
        (
            "localize",
            TaskKind.PERCEIVE,
            ["localization.pose_estimate"],
            "Observe localization and TF.",
        ),
        (
            "plan",
            TaskKind.CREATE_ARTIFACT,
            ["coverage.compute_path"],
            "Compute coverage through the selected component.",
        ),
        (
            "clean",
            TaskKind.REQUEST_ACTION,
            ["cleaning.enable"],
            "Enable cleaning through rosclawd.",
        ),
        (
            "execute",
            TaskKind.REQUEST_ACTION,
            ["coverage.execute"],
            "Execute coverage through rosclawd; observe obstacle handling.",
        ),
        (
            "verify",
            TaskKind.VALIDATE,
            ["coverage.verify"],
            "Independently verify cleaning footprint coverage and collisions.",
        ),
        (
            "recover",
            TaskKind.COORDINATE,
            ["coverage.compute_path", "coverage.execute"],
            "Queue missed reachable regions; wait for temporary blocks to clear; bounded retries through request_action.",
        ),
        (
            "disable",
            TaskKind.REQUEST_ACTION,
            ["cleaning.disable"],
            "Disable cleaning through rosclawd, including failure cleanup.",
        ),
        (
            "final",
            TaskKind.VALIDATE,
            ["coverage.verify"],
            "Require >=98% coverage, zero collisions and canonical receipts before completion.",
        ),
        (
            "remember",
            TaskKind.CREATE_ARTIFACT,
            [],
            "Save snapshot, diagnosis and verified outcome through Practice/Memory.",
        ),
    ]
    nodes: list[TaskNodeV1] = []
    for index, (name, kind, caps, goal) in enumerate(stages):
        nodes.append(
            TaskNodeV1(
                task_id=f"{mission_id}:{name}",
                mission_id=mission_id,
                kind=kind,
                goal=goal,
                dependencies=[nodes[-1].task_id] if nodes else [],
                required_capabilities=caps,
                inputs={"solution": solution} if index == 0 else {"snapshot_id": model.snapshot_id},
                constraints=TaskConstraints(
                    body_id=model.robot_id,
                    freshness_ms=5000,
                    risk_tier="HIGH" if kind == TaskKind.REQUEST_ACTION else "LOW",
                ),
                verification=TaskVerification(
                    verifier="deterministic:ros_coverage" if name in {"verify", "final"} else None
                ),
            )
        )
    graph = TaskGraphV1(mission_id=mission_id, nodes=nodes)
    graph.validate_dag()
    return graph
