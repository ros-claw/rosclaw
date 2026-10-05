"""Shared CLI/MCP read-only entrypoints with transport cleanup."""

from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.connectors.ros.probe import RosbridgeProbe
from rosclaw.connectors.ros.transport import RosbridgeEndpoint, RosbridgeTransport


def inspect_system(
    *, endpoint="ws://127.0.0.1:9090", robot_id="unknown", deep=False, body=None, event_bus=None
):
    transport = RosbridgeTransport(endpoint=RosbridgeEndpoint.from_url(endpoint), max_retries=0)
    try:
        model = RosbridgeProbe(transport).inspect(robot_id=robot_id, deep=deep, body=body)
        from rosclaw.connectors.ros.intelligence.evidence import emit_expert_evidence

        emit_expert_evidence(
            event_bus,
            "rosclaw.ros.system.snapshot.created",
            {"robot_id": robot_id, "snapshot_id": model.snapshot_id, "system": model.to_dict()},
        )
        return model
    finally:
        transport.close()


def load_system(data: dict) -> RosSystemModel:
    return RosSystemModel.from_dict(data)
