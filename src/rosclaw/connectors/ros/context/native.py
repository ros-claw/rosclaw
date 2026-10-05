"""Read fresh ROS observations for the existing Native Agent envelope."""

from datetime import UTC, datetime

from rosclaw.connectors.ros.context.body import configured_ros_body
from rosclaw.connectors.ros.context.compiler import compile_agent_summary
from rosclaw.connectors.ros.context.probe_client import read_snapshot as inspect_system
from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.resolver import resolve_capabilities


def native_ros_observations(service, mission, body) -> dict | None:
    configuration = (
        getattr(getattr(service, "_config", None), "raw", {}).get("agent", {}).get("ros_expert", {})
    )
    if not configuration.get("enabled", False):
        return None
    try:
        body_observations = {}
        if body:
            body_observations = configured_ros_body(
                mission.body_binding.body_id, body.effective_body_hash
            )
        model = inspect_system(
            endpoint=configuration.get("endpoint", "ws://127.0.0.1:9090"),
            robot_id=mission.body_binding.body_id,
            deep=True,
            body=body_observations,
        )
        now = datetime.now(UTC)
        if not -100 <= model.age_ms(now) <= 5000:
            raise ValueError("ROS observations are stale or future-dated")
        return {
            "status": "OBSERVED",
            "snapshot_id": model.snapshot_id,
            "snapshot_hash": model.snapshot_hash,
            "captured_at": model.captured_at.isoformat(),
            "age_ms": model.age_ms(now),
            "summary": compile_agent_summary(model, now=now),
            "admission_facts": {
                "status": "OBSERVED",
                "body_hash": model.body.get("effective_body_hash"),
                "readiness": {
                    c["semantic_id"]: c["status"] for c in resolve_capabilities(model, now=now)
                },
                "blocking_issues": sorted(
                    {
                        i["issue_code"]
                        for i in diagnose(model, now=now)["issues"]
                        if i["severity"] == "blocking"
                    }
                ),
            },
            "authorization": False,
        }
    except Exception as exc:
        return {"status": "UNKNOWN", "error": str(exc)[:300], "authorization": False}
