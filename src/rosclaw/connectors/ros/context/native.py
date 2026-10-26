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
        diagnosis = diagnose(model, now=now)
        from rosclaw.connectors.ros.resolver.semantics import issue_applies_to
        capabilities = resolve_capabilities(model, now=now)
        def hazard(issue):
            # Stable safety facts only: no capture IDs, prose or drifting ages.
            # Freshness changes are already reflected by readiness requirements.
            observation = issue.get("evidence", [{}])[0].get("observation", {})
            identity = {k: observation[k] for k in ("topic", "parent", "child", "name")
                        if isinstance(observation, dict) and k in observation}
            return {"code": issue["issue_code"], "identity": identity}
        blocking = [i for i in diagnosis["issues"] if i["severity"] == "blocking"]
        scopes = {c["semantic_id"]: {
            "readiness": c["status"],
            "requirements": c["requirements"],
            "interface": c["interface"],
            "blocking_issues": [hazard(i) for i in blocking if issue_applies_to(i, c["semantic_id"])],
        } for c in capabilities}
        return {
            "status": "OBSERVED",
            "diagnosis": diagnosis,
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
                "capability_facts": scopes,
                "global_hazards": [hazard(i) for i in blocking if i.get("affected_capabilities") is None or "*" in i.get("affected_capabilities", [])],
                "blocking_issues": sorted(
                    {
                        i["issue_code"]
                        for i in diagnosis["issues"]
                        if i["severity"] == "blocking"
                    }
                ),
            },
            "authorization": False,
        }
    except Exception as exc:
        return {"status": "UNKNOWN", "error": str(exc)[:300], "authorization": False}
