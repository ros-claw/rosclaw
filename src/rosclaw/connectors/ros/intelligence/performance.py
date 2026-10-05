"""Read-only performance topology; graph names never prove GPU placement."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from rosclaw.connectors.ros.intelligence.acceleration import acceleration_options
from rosclaw.connectors.ros.intelligence.system_model import RosSystemModel
from rosclaw.contracts.common import ContractModel, content_hash


class RosPerformanceGraph(ContractModel):
    SCHEMA = "rosclaw.ros_performance_graph.v1"
    schema_version: Literal["rosclaw.ros_performance_graph.v1"] = "rosclaw.ros_performance_graph.v1"
    snapshot_id: str
    captured_at: datetime
    nodes: list[dict]
    edges: list[dict]
    findings: list[dict]
    measured_copy_bytes: int | None
    validation_checks: dict[str, bool | None]
    optimization_verified: bool = False
    technology_options: list[dict] = Field(default_factory=list)
    graph_hash: str = ""


def performance_graph(model: RosSystemModel, *, now: datetime | None = None) -> RosPerformanceGraph:
    now = now or datetime.now(UTC)
    evidence = model.observations.get("performance", {})
    if not isinstance(evidence, dict):
        evidence = {}
    try:
        observed_at = datetime.fromisoformat(evidence["captured_at"])
        fresh = (
            bool(evidence.get("source"))
            and observed_at.tzinfo is not None
            and -0.1 <= (now - observed_at).total_seconds() <= 5
            and -100 <= model.age_ms(now) <= 5000
        )
    except (KeyError, TypeError, ValueError):
        fresh = False
    measurements = evidence.get("nodes", {}) if fresh else {}
    if not isinstance(measurements, dict):
        measurements = {}
    nodes = []
    placements = {}
    for node in model.graph.get("nodes", []):
        name = node["name"]
        measured = measurements.get(name, {})
        if not isinstance(measured, dict):
            measured = {}
        backend = measured.get("buffer_backend", "UNKNOWN")
        if backend not in {"CPU", "CUDA"} or not measured.get("source"):
            backend = "UNKNOWN"
        pid = measured.get("pid")
        if type(pid) is not int or pid <= 0:
            pid = None
        item = {
            "name": name,
            "buffer_backend": backend,
            "pid": pid,
            "source": measured.get("source"),
        }
        nodes.append(item)
        placements[name] = item
    edges, findings = [], []
    for topic in model.graph.get("topics", []):
        for publisher in sorted(set(topic.get("publishers", []))):
            for subscriber in sorted(set(topic.get("subscribers", []))):
                source, target = placements.get(publisher, {}), placements.get(subscriber, {})
                backends = [
                    source.get("buffer_backend", "UNKNOWN"),
                    target.get("buffer_backend", "UNKNOWN"),
                ]
                boundary = None
                if source.get("pid") and target.get("pid"):
                    boundary = source["pid"] != target["pid"]
                edge = {
                    "topic": topic["name"],
                    "publisher": publisher,
                    "subscriber": subscriber,
                    "buffer_backends": backends,
                    "separate_process": boundary,
                    "zero_copy_verified": False,
                }
                edges.append(edge)
                if set(backends) == {"CPU", "CUDA"}:
                    findings.append(
                        {
                            "code": "CPU_CUDA_BOUNDARY",
                            "evidence_class": "inferred",
                            "edge": edge,
                            "next_check": "Measure payload-sized copies in a trace; graph placement alone does not prove a copy.",
                        }
                    )
    copy_bytes = 0 if fresh and evidence.get("copy_trace_complete") is True else None
    if fresh:
        trace_valid = True
        events = evidence.get("copy_events", [])
        if not isinstance(events, list):
            events = []
            copy_bytes = None
            trace_valid = False
        for event in events:
            if (
                not isinstance(event, dict)
                or event.get("direction") not in {"HtoD", "DtoH", "DtoD", "HtoH"}
                or not isinstance(event.get("source"), str)
                or not event["source"].strip()
                or type(event.get("bytes")) is not int
                or event["bytes"] <= 0
            ):
                copy_bytes = None
                trace_valid = False
                continue
            if (
                event.get("direction") in {"HtoD", "DtoH"}
                and event.get("source")
                and type(event.get("bytes")) is int
                and event["bytes"] > 0
            ):
                copy_bytes = (copy_bytes or 0) + event["bytes"]
                findings.append(
                    {
                        "code": "MEASURED_HOST_DEVICE_COPY",
                        "evidence_class": "observed",
                        "event": event,
                    }
                )
        if not trace_valid:
            copy_bytes = None
    observed_checks = evidence.get("validation_checks", {}) if fresh else {}
    if not isinstance(observed_checks, dict):
        observed_checks = {}
    checks = {
        key: observed_checks.get(key)
        for key in (
            "backend_type",
            "separate_process_transport",
            "cpu_fallback",
            "buffer_lifetime",
            "memory_copy",
        )
    }
    checks = {key: value if type(value) is bool else None for key, value in checks.items()}
    # A single topology/trace is never a before/after optimization benchmark.
    result = RosPerformanceGraph(
        snapshot_id=model.snapshot_id,
        captured_at=model.captured_at,
        nodes=nodes,
        edges=edges,
        findings=findings,
        measured_copy_bytes=copy_bytes,
        validation_checks=checks,
        technology_options=acceleration_options(model),
    )
    payload = result.model_dump(mode="json")
    payload.pop("graph_hash")
    result.graph_hash = content_hash("rosperf", payload)
    return result
