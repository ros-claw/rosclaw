"""Compile sealed observations to a bounded graph; never collect or execute ROS I/O."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime

from rosclaw.connectors.ros.engineering.case_compiler import compile_engineering_case
from rosclaw.connectors.ros.intelligence.system_model import RosSystemModel
from rosclaw.contracts.common import content_hash

from .graph_contracts import (
    OperationalDataflowV1,
    OperationalEndpointV1,
    OperationalEntityV1,
    OperationalTimeV1,
    OperationalTransformV1,
    RosConstraintFindingV1,
    RosOperationalGraphV1,
)
from .provenance_edges import freshness, provenance_edge
from .tf_authority import current_tf_attributions


def _id(kind, value):
    return content_hash("op_" + kind, value)


def _integer(value):
    return value if type(value) is int and value > 0 else None


def _policy_status(pub, sub):
    """Directional check of only observed policies; defaults stay UNKNOWN."""
    states = []
    for key, levels in (
        ("reliability", ["BEST_EFFORT", "RELIABLE"]),
        ("durability", ["VOLATILE", "TRANSIENT_LOCAL"]),
        ("liveliness", ["AUTOMATIC", "MANUAL_BY_TOPIC"]),
    ):
        offered, requested = getattr(pub, key), getattr(sub, key)
        states.append(
            None
            if offered not in levels or requested not in levels
            else levels.index(offered) >= levels.index(requested)
        )
    offered, requested = pub.deadline_ns, sub.deadline_ns
    states.append(None if offered is None or requested is None else offered <= requested)
    return "VIOLATED" if False in states else "UNKNOWN" if None in states else "SATISFIED"


def compile_operational_graph(
    model: RosSystemModel,
    urdf_bytes: bytes | None = None,
    requirements: dict | None = None,
    *,
    source_ref: str,
    now: datetime,
    evidence_domain="SEALED_REPLAY",
    source_review=None,
    expected_run_id=None,
    expected_body_hash=None,
) -> RosOperationalGraphV1:
    """Historical replays require explicit original clocks; all results are diagnostic.

    Reject excessive input rather than silently truncate a graph and call it complete.
    Existing five-mechanism invariants are copied from the unchanged Case compiler.
    The graph adds scoped endpoint/time/authority findings, not repair templates.
    """
    if not source_ref or len(source_ref) > 2048 or now.tzinfo is None:
        raise ValueError("bounded source reference and aware explicit review clock required")
    native_only = urdf_bytes is None and requirements is None
    if (urdf_bytes is None) != (requirements is None):
        raise ValueError("navigation context requires both original URDF and requirements")
    if native_only and any(
        value is not None for value in (source_review, expected_run_id, expected_body_hash)
    ):
        raise ValueError("native observations cannot assert independent Body/source admission")
    if not native_only and (type(urdf_bytes) is not bytes or not 0 < len(urdf_bytes) <= 5_000_000):
        raise ValueError("bounded nonempty original URDF bytes required")
    for name, document in (("requirements", requirements), ("source review", source_review)):
        if document is None and (name == "source review" or native_only):
            continue
        if type(document) is not dict:
            raise ValueError("bounded original " + name + " object required")
        if len(json.dumps(document, ensure_ascii=False, allow_nan=False).encode()) > 5_000_000:
            raise ValueError("original " + name + " exceeds 5MB")
    model = RosSystemModel.from_dict(model.to_dict())
    snapshot = model.to_dict()
    if len(json.dumps(snapshot, ensure_ascii=False).encode()) > 5_000_000:
        raise ValueError("operational graph input exceeds 5MB")
    if any(len(snapshot.get(key, [])) > 1024 for key in ("signals", "transforms", "lifecycle")):
        raise ValueError("bounded observation collection required")
    for key in ("topics", "nodes", "services", "actions"):
        if type(model.graph.get(key, [])) is not list or len(model.graph.get(key, [])) > 1024:
            raise ValueError("bounded native graph lists required")
    raw_endpoints = model.qos.get("endpoints", [])
    if type(raw_endpoints) is not list or len(raw_endpoints) > 1024:
        raise ValueError("bounded native endpoint list required")
    case = (
        None
        if native_only
        else compile_engineering_case(
            model,
            urdf_bytes,
            requirements,
            source_ref=source_ref,
            now=now,
            evidence_domain=evidence_domain,
            source_review=source_review,
            expected_run_id=expected_run_id,
            expected_body_hash=expected_body_hash,
        )
    )
    case_doc = case.model_dump(mode="json") if case else None
    source_status = case.source_status if case else "UNKNOWN"
    max_age = 500
    snapshot_freshness = freshness(model.captured_at, now, max_age)
    usable = snapshot_freshness == "FRESH" and source_status != "MISMATCH"
    provenance = {}
    entities = {}
    endpoints = []
    flows = []
    transforms = []
    time_facts = []
    findings = []

    def ref(pointer, *, stamp=None, derived=False):
        if derived and case is None:
            raise ValueError("no derived navigation Case in native observation mode")
        record = provenance_edge(
            case_doc if derived else snapshot,
            case.case_hash if derived else source_ref,
            case.case_hash if derived else model.snapshot_hash,
            pointer,
            stamp or model.captured_at,
            now,
            max_age,
        )
        provenance[record.id] = record
        return record.id

    def entity(kind, name, refs, interface_type=None):
        if not isinstance(name, str) or not 0 < len(name) <= (2048 if kind == "source" else 512):
            raise ValueError("bounded explicit graph entity name required")
        if interface_type is not None and (
            not isinstance(interface_type, str) or len(interface_type) > 512
        ):
            raise ValueError("bounded interface type required")
        key = _id("entity", [kind, name])
        if key not in entities:
            entities[key] = OperationalEntityV1(
                id=key,
                kind=kind,
                name=name,
                interface_type=interface_type,
                evidence_refs=list(refs),
            )
        else:
            old = entities[key]
            # Conflicting declared types are retained as unknown, never pick the first.
            dtype = (
                old.interface_type
                if interface_type is None or old.interface_type == interface_type
                else interface_type
                if old.interface_type is None
                else None
            )
            entities[key] = old.model_copy(
                update={
                    "evidence_refs": sorted(set(old.evidence_refs + list(refs))),
                    "interface_type": dtype,
                }
            )
        return key

    def finding(key, invariant, status, scope, refs, targets=(), missing=()):
        findings.append(
            RosConstraintFindingV1(
                id=key,
                invariant=invariant,
                status=status if usable else "UNKNOWN",
                scope=scope,
                evidence_refs=list(refs),
                targets=list(targets),
                missing_evidence=list(missing),
            )
        )

    if case is not None:
        source_fact = ref("/source_status", derived=True)
        entity("source", source_ref, [source_fact])
        entity("urdf", case.source_urdf_sha256, [ref("/source_urdf_sha256", derived=True)])
        if case.run_id:
            entity("run", case.run_id, [ref("/run_id", derived=True)])
        if case.body_hash:
            entity("body", case.body_hash, [ref("/body_hash", derived=True)])
        source_document = case.derived_documents["source_review"]
        if isinstance(source_document.get("world_id"), str) and source_document["world_id"]:
            entity(
                "world",
                source_document["world_id"],
                [ref("/derived_documents/source_review/world_id", derived=True)],
            )
        legacy_targets = {
            "F-SOURCE-001": [source_ref],
            "F-TF-001": [requirements["lifecycle_nodes"]["localization"]],
            "F-SCAN-001": [requirements["interface_names"].get("sensing.lidar", source_ref)],
            "F-LIFE-001": sorted(requirements["lifecycle_nodes"].values()),
            "F-GEOM-001": list(requirements["costmap_topics"]),
            "F-ROUTE-001": [requirements["velocity_output_topic"]],
        }
        for i, constraint in enumerate(case.constraints):
            # Keep exactly the original Case verdict, including source mismatch itself.
            findings.append(
                RosConstraintFindingV1(
                    id=constraint.id,
                    invariant=constraint.invariant,
                    status=constraint.status,
                    scope="EXISTING_CASE_INVARIANT_UNCHANGED",
                    targets=legacy_targets.get(constraint.id, []),
                    evidence_refs=[ref(f"/constraints/{i}", derived=True)],
                    missing_evidence=constraint.next_checks
                    if constraint.status == "UNKNOWN"
                    else [],
                )
            )
    else:
        source_fact = ref("/snapshot_hash")
        entity("source", source_ref, [source_fact])
        finding(
            "F-SOURCE-001",
            "source.identity_matches",
            "UNKNOWN",
            "NATIVE_OBSERVATIONS_WITHOUT_BODY_OR_NAVIGATION_CONTEXT",
            [source_fact],
            [source_ref],
            ["independent_current_source_run_World_and_immutable_Body_binding"],
        )
    for collection, kind, type_key in (
        ("nodes", "node", None),
        ("topics", "topic", "msg_type"),
        ("services", "service", "srv_type"),
        ("actions", "action", "action_type"),
    ):
        for i, row in enumerate(model.graph.get(collection, [])):
            if type(row) is not dict or not isinstance(row.get("name"), str) or not row["name"]:
                raise ValueError("native graph rows require explicit names")
            entity(
                kind,
                row["name"],
                [ref(f"/graph/{collection}/{i}")],
                row.get(type_key) or None if type_key else None,
            )

    signatures = Counter(
        (e.get("topic"), e.get("node"), e.get("kind")) for e in raw_endpoints if type(e) is dict
    )
    gids = Counter(e.get("gid") for e in raw_endpoints if type(e) is dict and e.get("gid"))
    for i, row in enumerate(raw_endpoints):
        if (
            type(row) is not dict
            or row.get("kind") not in {"publisher", "subscriber"}
            or not isinstance(row.get("node"), str)
            or not row["node"].startswith("/")
            or not isinstance(row.get("topic"), str)
            or not row["topic"].startswith("/")
        ):
            raise ValueError(
                "native endpoints require kind and fully-qualified actual node/topic names"
            )
        fact = ref(f"/qos/endpoints/{i}")
        topic = entity("topic", row["topic"], [fact])
        node = entity("node", row["node"], [fact])
        gid = row.get("gid") if isinstance(row.get("gid"), str) and row["gid"] else None
        ambiguous = (gid is not None and gids[gid] != 1) or (
            gid is None and signatures[row["topic"], row["node"], row["kind"]] > 1
        )
        identity = "AMBIGUOUS" if ambiguous else "OBSERVED_GID" if gid else "NODE_ONLY"
        ep = OperationalEndpointV1(
            id=_id(
                "endpoint", [model.snapshot_hash, i, row["topic"], row["node"], row["kind"], gid]
            ),
            topic_id=topic,
            node_id=node,
            kind=row["kind"],
            gid=gid,
            identity_status=identity,
            reliability=row.get("reliability"),
            durability=row.get("durability"),
            liveliness=row.get("liveliness"),
            deadline_ns=_integer(row.get("deadline_ns")),
            evidence_refs=[fact],
        )
        endpoints.append(ep)
    # Topic grouping prevents unrelated namespaces/remaps from accidentally matching.
    grouped = {}
    for endpoint in endpoints:
        grouped.setdefault(endpoint.topic_id, {"publisher": [], "subscriber": []})[
            endpoint.kind
        ].append(endpoint)
    pair_count = sum(len(g["publisher"]) * len(g["subscriber"]) for g in grouped.values())
    if pair_count > 4096:
        raise ValueError("operational dataflow pair budget exceeded")
    for topic, group in sorted(grouped.items()):
        topic_name = entities[topic].name
        for pub in group["publisher"]:
            for sub in group["subscriber"]:
                status = _policy_status(pub, sub)
                if not usable or "AMBIGUOUS" in {pub.identity_status, sub.identity_status}:
                    status = "UNKNOWN"
                flow_id = _id("flow", [pub.id, sub.id])
                refs = pub.evidence_refs + sub.evidence_refs
                flows.append(
                    OperationalDataflowV1(
                        id=flow_id,
                        publisher_id=pub.id,
                        subscriber_id=sub.id,
                        observed_policy_status=status,
                        evidence_refs=refs,
                    )
                )
                finding(
                    _id("finding", flow_id),
                    "qos.observed_requested_offered_policies",
                    status,
                    "RELIABILITY_DURABILITY_LIVELINESS_DEADLINE_ONLY_NOT_DELIVERY",
                    refs,
                    [topic_name, entities[pub.node_id].name, entities[sub.node_id].name],
                    ["resolved_endpoint_GIDs_and_full_RMW_compatibility_including_lease_duration"],
                )
    qos_refs = [ref("/qos")]
    finding(
        "OP-QOS-COVERAGE",
        "qos.complete_paired_endpoints",
        "UNKNOWN",
        "FULL_COMPATIBILITY_NOT_ESTABLISHED_BY_PARTIAL_FIELDS",
        qos_refs,
        sorted({entities[e.topic_id].name for e in endpoints}),
        ["complete_source_pinned_endpoint_QoS_and_native_compatibility_result"],
    )

    domains = model.observations.get("node_use_sim_time", {})
    if type(domains) is dict and len(domains) > 1024:
        raise ValueError("bounded per-node clock domain mapping required")
    domain_values = list(domains.values()) if type(domains) is dict else []
    resolved = bool(domain_values) and all(type(v) is bool for v in domain_values)
    aligned = resolved and len(set(domain_values)) == 1
    clock_domain = (
        "ROS_SIM" if aligned and domain_values[0] else "ROS_SYSTEM" if aligned else "UNKNOWN"
    )
    domain_refs = (
        [ref("/observations/node_use_sim_time")]
        if "node_use_sim_time" in model.observations
        else [ref("/observations")]
    )
    parameter_stamps = model.observations.get("parameter_captured_at", {})
    parameter_values = model.observations.get("node_parameters", {})
    if type(parameter_stamps) is not dict or len(parameter_stamps) > 1024:
        raise ValueError("bounded per-node parameter capture times required")
    domain_current = usable and model.completeness.get("time") is True and resolved
    for node in domains if type(domains) is dict else []:
        if not isinstance(node, str) or not 0 < len(node) <= 512:
            raise ValueError("bounded clock node names required")
        raw_stamp = parameter_stamps.get(node)
        values = parameter_values.get(node, {}) if type(parameter_values) is dict else {}
        if (
            type(values) is not dict
            or type(values.get("use_sim_time")) is not bool
            or values["use_sim_time"] is not domains[node]
        ):
            domain_current = False
        try:
            stamp = datetime.fromisoformat(raw_stamp) if isinstance(raw_stamp, str) else None
        except ValueError:
            stamp = None
        if stamp is None or stamp.tzinfo is None:
            domain_current = False
            stamp = model.captured_at  # identifies the value, not a fresh RPC claim
        elif freshness(stamp, now, max_age) != "FRESH":
            domain_current = False
        pointer_node = node.replace("~", "~0").replace("/", "~1")
        domain_refs.append(ref("/observations/node_use_sim_time/" + pointer_node, stamp=stamp))
        if type(parameter_values) is dict and node in parameter_values:
            domain_refs.append(ref("/observations/node_parameters/" + pointer_node, stamp=stamp))
    if "parameter_captured_at" in model.observations:
        domain_refs.append(ref("/observations/parameter_captured_at"))
    declared_clock_domain = clock_domain
    clock_refs = list(domain_refs)
    probe_clock = model.environment.get("probe_use_sim_time")
    clock_refs.append(
        ref("/environment/probe_use_sim_time")
        if "probe_use_sim_time" in model.environment
        else ref("/environment")
    )
    receive_age = model.observations.get("clock_last_receive_age_ms")
    if "clock_last_receive_age_ms" in model.observations:
        clock_refs.append(ref("/observations/clock_last_receive_age_ms"))
    if "clock_advancing" in model.observations:
        clock_refs.append(ref("/observations/clock_advancing"))
    clock_current = domain_current and (
        declared_clock_domain == "ROS_SYSTEM"
        and probe_clock is False
        or declared_clock_domain == "ROS_SIM"
        and type(receive_age) in (int, float)
        and math.isfinite(receive_age)
        and 0 <= receive_age + (now - model.captured_at).total_seconds() * 1000 <= max_age
    )
    if not clock_current:
        clock_domain = "UNKNOWN"
    clock_state = "SATISFIED" if aligned else "VIOLATED" if resolved and not aligned else "UNKNOWN"
    finding(
        "OP-TIME-DOMAINS",
        "time.observed_node_clock_domains_agree",
        clock_state if domain_current else "UNKNOWN",
        "OBSERVED_NODES_ONLY_NOT_GLOBAL_CLOCK_SYNCHRONIZATION",
        domain_refs,
        sorted(domains) if type(domains) is dict else [],
        []
        if aligned and domain_current
        else ["same_capture_per_node_use_sim_time_ROS_clock_and_steady_receive_window"],
    )
    if declared_clock_domain == "ROS_SIM":
        advancing = model.observations.get("clock_advancing")
        finding(
            "OP-CLOCK-PROGRESS",
            "time.observed_sim_clock_advances",
            "SATISFIED"
            if clock_current and advancing is True
            else "VIOLATED"
            if clock_current and advancing is False
            else "UNKNOWN",
            "BOUNDED_OBSERVATION_WINDOW_NOT_FUTURE_GUARANTEE",
            clock_refs,
            ["observed_ROS_clock"],
            ["ROS_clock_samples_with_independent_steady_receive_times"]
            if advancing is not True
            else [],
        )
    authority_groups = {}
    for i, edge in enumerate(model.transforms):
        fact = ref(f"/transforms/{i}", stamp=edge.captured_at)
        parent = entity("frame", edge.parent, [fact])
        child = entity("frame", edge.child, [fact])
        authority = getattr(edge, "authority", None)
        authority = authority if isinstance(authority, str) and authority else None
        if authority is not None and len(authority) > 512:
            raise ValueError("bounded TF authority name required")
        transforms.append(
            OperationalTransformV1(
                id=_id("tf", [model.snapshot_hash, i]),
                parent_id=parent,
                child_id=child,
                static=edge.static,
                authority=authority,
                age_ms=edge.age_ms,
                clock_domain=clock_domain,
                evidence_refs=[fact, *clock_refs],
            )
        )
        edge_current = (
            usable
            and provenance[fact].freshness == "FRESH"
            and model.completeness.get("tf") is True
        )
        # Do not compare ROS stamps to wall/steady or assert ages in mixed/unknown domains.
        time_status = "UNKNOWN"
        if (
            edge_current
            and clock_current
            and aligned
            and not edge.static
            and edge.age_ms is not None
        ):
            time_status = (
                "SATISFIED" if -edge.future_tolerance_ms <= edge.age_ms <= max_age else "VIOLATED"
            )
        if not edge.static:
            finding(
                _id("tf_time", [model.snapshot_hash, i]),
                "tf.observed_dynamic_time_window",
                time_status,
                "DYNAMIC_ROS_CLOCK_AGE_WITH_DECLARED_FUTURE_TOLERANCE",
                [fact, *clock_refs],
                [edge.parent, edge.child],
                []
                if time_status != "UNKNOWN"
                else ["current_TF_stamp_clock_domain_and_steady_receive_age"],
            )
        authority_groups.setdefault(edge.child, []).append(
            (edge.parent, authority, fact, edge_current)
        )
    attributed_groups = {}
    for index, row, stamp in current_tf_attributions(model, now=now):
        fact = ref(f"/observations/tf_attribution_window/records/{index}", stamp=stamp)
        entity("frame", row["parent"], [fact])
        entity("frame", row["child"], [fact])
        attributed_groups.setdefault(
            (
                model.observations["tf_attribution_window"]["observer_instance"],
                row["reader_scope"],
                row["topic"],
                row["child"],
            ),
            [],
        ).append((row["parent"], row["publisher_token"], fact))
    for (observer_instance, reader_scope, topic, child), rows in sorted(attributed_groups.items()):
        conflict = len({r[0] for r in rows}) > 1 or len({r[1] for r in rows}) > 1
        finding(
            _id("message_authority", [observer_instance, reader_scope, topic, child]),
            "tf.observed_message_authorities_per_reader",
            "VIOLATED" if usable and conflict else "UNKNOWN",
            "THIS_OBSERVER_READER_TOPIC_ONLY_NOT_NODE_ID_OR_GLOBAL_GID_OR_UNIQUENESS",
            [r[2] for r in rows],
            [topic, child],
            [] if conflict else ["complete_source_pinned_consumer_TF_buffer_authority_review"],
        )
    if not authority_groups:
        finding(
            "OP-TF-AUTHORITY-MISSING",
            "tf.single_parent_and_authority",
            "UNKNOWN",
            "NO_ATTRIBUTED_TF_WINDOW",
            [ref("/transforms")],
            missing=["complete_TF_window_with_publisher_GID_authority_and_parent_child"],
        )
    for child, rows in sorted(authority_groups.items()):
        parents = {r[0] for r in rows}
        authorities = {r[1] for r in rows if r[1]}
        current = all(r[3] for r in rows)
        # A witnessed contradiction can violate with partial coverage; uniqueness cannot.
        state = "VIOLATED" if current and (len(parents) > 1 or len(authorities) > 1) else "UNKNOWN"
        finding(
            _id("authority", child),
            "tf.single_parent_and_authority",
            state,
            "MISSING_PUBLISHER_ATTRIBUTION_OR_COMPLETE_WINDOW_NEVER_PROVES_UNIQUENESS",
            [r[2] for r in rows],
            [child],
            ["complete_TF_window_with_publisher_GID_authority_and_parent_child"]
            if state == "UNKNOWN"
            else [],
        )
    for i, signal in enumerate(model.signals):
        fact = ref(f"/signals/{i}", stamp=signal.captured_at)
        entity("topic", signal.topic, [fact])
        time_facts.append(
            OperationalTimeV1(
                id=_id("signal_time", [model.snapshot_hash, i]),
                subject=signal.topic,
                domain="UNKNOWN",
                receive_clock="STEADY" if "monotonic" in signal.source else "UNKNOWN",
                sample_rate_hz=signal.rate_hz,
                receive_age_ms=signal.last_message_age_ms,
                evidence_refs=[fact],
            )
        )
        if native_only:
            current = (
                usable
                and provenance[fact].freshness == "FRESH"
                and model.completeness.get("signals") is True
                and "monotonic" in signal.source
                and getattr(signal, "freshness_policy", "stream") != "latched"
            )
            age = signal.last_message_age_ms
            status = "UNKNOWN"
            if current and age is not None:
                cumulative_age = age + (now - signal.captured_at).total_seconds() * 1000
                status = (
                    "SATISFIED" if cumulative_age <= min(max_age, signal.max_age_ms) else "VIOLATED"
                )
            finding(
                _id("receive_window", [model.snapshot_hash, i]),
                "signal.observed_steady_receive_window",
                status,
                "THIS_OBSERVER_ONLY_NOT_CONSUMER_DELIVERY_OR_PRODUCER_ROOT_CAUSE",
                [fact],
                [signal.topic],
                ["current_steady_receive_window_with_nonlatched_signal_age"]
                if status == "UNKNOWN"
                else [],
            )
    message_frames = model.observations.get("message_frames", {})
    if type(message_frames) is not dict or len(message_frames) > 1024:
        raise ValueError("bounded message header mapping required")
    for topic, header in sorted(message_frames.items()):
        if type(header) is not dict:
            raise ValueError("message header observation must be an object")
        pointer = "/observations/message_frames/" + topic.replace("~", "~0").replace("/", "~1")
        stamp = header.get("captured_at")
        if not stamp:
            continue
        fact = ref(pointer, stamp=stamp)
        time_facts.append(
            OperationalTimeV1(
                id=_id("message_time", [model.snapshot_hash, topic]),
                subject=topic,
                domain=clock_domain
                if clock_current and provenance[fact].freshness == "FRESH"
                else "UNKNOWN",
                message_age_ms=header.get("stamp_age_ms"),
                evidence_refs=[fact, *clock_refs],
            )
        )
    for i, lifecycle in enumerate(model.lifecycle):
        entity("node", lifecycle.name, [ref(f"/lifecycle/{i}", stamp=lifecycle.captured_at)])
    payload = {
        "source_ref": source_ref,
        "snapshot_hash": model.snapshot_hash,
        "case_hash": case.case_hash if case else None,
        "source_urdf_sha256": case.source_urdf_sha256 if case else None,
        "requirements_hash": content_hash("navrequirements", requirements)
        if requirements is not None
        else None,
        "reviewed_at": now,
        "max_age_ms": max_age,
        "evidence_domain": evidence_domain,
        "snapshot_freshness": snapshot_freshness,
        "source_status": source_status,
        "entities": sorted(entities.values(), key=lambda x: x.id),
        "endpoints": endpoints,
        "dataflows": flows,
        "transforms": transforms,
        "time_facts": time_facts,
        "findings": findings,
        "provenance": provenance,
        "completeness": dict(model.completeness),
    }
    graph = RosOperationalGraphV1(graph_hash="", **payload)
    data = graph.model_dump(mode="json")
    data.pop("graph_hash")
    return RosOperationalGraphV1.from_dict({**data, "graph_hash": content_hash("rosopgraph", data)})


def summarize_operational_graph(graph: RosOperationalGraphV1, *, max_characters=6000):
    """Bounded Agent projection with explicit omitted counts, never a complete graph."""
    if type(max_characters) is not int or not 1200 <= max_characters <= 12000:
        raise ValueError("summary budget requires 1200..12000 characters")
    graph = RosOperationalGraphV1.from_dict(graph.model_dump(mode="json"))
    summary = {
        "graph_hash": graph.graph_hash,
        "snapshot_hash": graph.snapshot_hash,
        "case_hash": graph.case_hash,
        "scope": "BOUNDED_PROJECTION_ORIGINAL_GRAPH_REQUIRED_FOR_FULL_EVIDENCE",
        "current_authority": "NONE",
        "current_l1": "NOT_EVALUATED",
        "evidence_domain": graph.evidence_domain,
        "freshness": graph.snapshot_freshness,
        "source_status": graph.source_status,
        "finding_counts": dict(Counter(f.status for f in graph.findings)),
        "total_findings": len(graph.findings),
        "omitted_findings": len(graph.findings),
        "findings": [],
    }
    priority = {"VIOLATED": 0, "UNKNOWN": 1, "SATISFIED": 2}
    for finding in sorted(graph.findings, key=lambda f: (priority[f.status], f.id)):
        row = {"id": finding.id, "invariant": finding.invariant, "status": finding.status}
        summary["findings"].append(row)
        summary["omitted_findings"] -= 1
        if len(json.dumps(summary, ensure_ascii=False, separators=(",", ":"))) > max_characters:
            summary["findings"].pop()
            summary["omitted_findings"] += 1
            break
    return summary
