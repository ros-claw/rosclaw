"""Pure task constraint differences; no I/O, ROS transport or runtime imports."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import TypeGuard, cast

from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.diagnosis.engine import frame_connected
from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.contracts.common import content_hash

from .contracts import ConstraintDeltaV1, EvidenceRefV1, RepairIntentV1, RosEngineeringCaseV1


def _pointer(*parts):
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def _fresh(stamp, now):
    try:
        age = (now - datetime.fromisoformat(stamp)).total_seconds() * 1000
        return 0 <= age <= 500
    except (ValueError, TypeError):
        return False


def _number(value: object) -> TypeGuard[int | float]:
    return type(value) in (int, float) and math.isfinite(cast(float, value))


def compile_engineering_case(
    model: RosSystemModel,
    urdf_bytes: bytes,
    requirements: dict,
    *,
    source_ref: str,
    now: datetime | None = None,
    evidence_domain="OBSERVATION",
    source_review=None,
    body=None,
    expected_run_id=None,
    expected_body_hash=None,
) -> RosEngineeringCaseV1:
    """Describe measured constraints and conditional proposals, never a live gate.

    SEALED_REPLAY requires an explicit original review clock; replaying at that
    clock is historical diagnosis, not evidence of readiness at the current time.
    The caller supplies original source/Body review objects, never a case label.
    """
    if evidence_domain not in {"OBSERVATION", "SEALED_REPLAY"} or not source_ref:
        raise ValueError("explicit evidence domain and stable original source ref required")
    if evidence_domain == "SEALED_REPLAY" and now is None:
        raise ValueError("replay requires original review clock")
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("aware review clock required")
    if model.snapshot_hash != model.compute_snapshot_hash():
        raise ValueError("sealed model hash mismatch")
    review = review_navigation_candidate(model, urdf_bytes, requirements, now=now)
    diagnosis = diagnose(model, profile="navigation", now=now, max_age_ms=500)
    snapshot = model.to_dict()
    index = {}
    constraints = []
    repairs = []
    source = source_review or {}
    source_status = source.get("result", "UNKNOWN")
    if source_status not in {"MATCHED", "MISMATCH", "UNKNOWN"}:
        source_status = "UNKNOWN"
    run_id = source.get("sim_run_id")
    body_hash = source.get("body_hash")
    if source_status == "MATCHED" and not (
        source.get("schema_version") == "rosclaw.sim_source_binding_evidence.v1"
        and source.get("ros_runtime_snapshot_hash") == model.snapshot_hash
        and _fresh(source.get("reviewed_at"), now)
        and _fresh(source.get("observed_at"), now)
        and run_id
        and body_hash
    ):
        source_status = "UNKNOWN"
    if (expected_run_id is not None and run_id is not None and run_id != expected_run_id) or (
        expected_body_hash is not None and body_hash is not None and body_hash != expected_body_hash
    ):
        source_status = "MISMATCH"
    if source.get("ros_runtime_snapshot_hash") not in {None, model.snapshot_hash}:
        source_status = "MISMATCH"
    derived_documents = {"source_review": source, "navigation_observation_review": review}

    if body is not None:
        if body.compute_hash() != body.effective_body_hash:
            source_status = "MISMATCH"
        if body_hash is not None and body_hash != body.effective_body_hash:
            source_status = "MISMATCH"
        body_hash = body.effective_body_hash
    snapshot_current = (
        _fresh(model.captured_at.isoformat(), now) and model.completeness.get("graph") is True
    )
    usable = snapshot_current and source_status != "MISMATCH"

    def evidence(pointer, value, stamp=None, classification="MEASURED"):
        document = snapshot
        ref = source_ref
        if pointer == "/source_review":
            document = source
            pointer = ""
            ref = content_hash("derived", source)
        elif pointer.startswith("/candidate/"):
            document = review
            ref = content_hash("derived", review)
        try:
            original_value = document
            for part in pointer.split("/")[1:] if pointer else []:
                key_part = part.replace("~1", "/").replace("~0", "~")
                original_value = (
                    original_value[int(key_part)]
                    if isinstance(original_value, list)
                    else original_value[key_part]
                )
        except (KeyError, IndexError, TypeError, ValueError):
            original_value = None
            classification = "UNKNOWN"
        if not usable:
            classification = "UNKNOWN"
        key = (
            "fact_"
            + content_hash(
                "rosfact", {"pointer": pointer, "document": ref, "snapshot": model.snapshot_hash}
            ).rsplit("_", 1)[1][:16]
        )
        index[key] = EvidenceRefV1(
            source_ref=ref,
            snapshot_hash=model.snapshot_hash,
            pointer=pointer,
            observed_at=stamp or model.captured_at.isoformat(),
            value_hash=content_hash("factvalue", original_value),
            classification=classification,
        )
        return key

    def add(code, invariant, status, expected, observed, refs, checks=(), hypotheses=()):
        constraints.append(
            ConstraintDeltaV1(
                id=code,
                invariant=invariant,
                status=status if usable else "UNKNOWN",
                expected=expected,
                observed=observed,
                evidence_refs=refs,
                next_checks=list(checks),
                hypotheses=list(hypotheses),
            )
        )

    def propose(code, field, value, refs, conditions):
        if usable:
            repairs.append(
                RepairIntentV1(
                    id=code,
                    semantic_field=field,
                    proposed_value=value,
                    evidence_refs=refs,
                    preconditions=conditions,
                )
            )

    source_fact = evidence(
        "/source_review",
        source or None,
        classification="UNKNOWN" if source_status == "UNKNOWN" else "MEASURED",
    )
    constraints.append(
        ConstraintDeltaV1(
            id="F-SOURCE-001",
            invariant="source.identity_matches",
            status="VIOLATED"
            if source_status == "MISMATCH"
            else "SATISFIED"
            if source_status == "MATCHED" and usable
            else "UNKNOWN",
            expected="independently matched current source, run and immutable Body",
            observed=source_status,
            evidence_refs=[source_fact],
            next_checks=["review_original_source_binding_and_current_continuity"],
        )
    )
    frames = review["candidate"]["frames"]
    map_frame = frames.get("map")
    odom = frames.get("odom")
    connected = frame_connected(model, map_frame, odom) if map_frame and odom else None
    edges = [e for e in model.transforms if e.parent == map_frame and e.child == odom]
    edge_fresh = bool(edges) and all(
        not e.static
        and _fresh(e.captured_at.isoformat(), now)
        and _number(e.age_ms)
        and 0 <= e.age_ms <= 500
        for e in edges
    )
    tf_status = (
        "VIOLATED"
        if connected is False
        else "SATISFIED"
        if connected and edge_fresh and len(edges) == 1
        else "UNKNOWN"
    )
    tf_ref = evidence("/transforms", [e.model_dump(mode="json") for e in model.transforms])
    loc = requirements["lifecycle_nodes"]["localization"]
    params = model.observations.get("node_parameters", {})
    stamps = model.observations.get("parameter_captured_at", {})
    amcl = params.get(loc, {})
    broadcast = amcl.get("tf_broadcast")
    amcl_ref = evidence(
        _pointer("observations", "node_parameters", loc, "tf_broadcast"),
        broadcast,
        stamps.get(loc),
        "MEASURED" if _fresh(stamps.get(loc), now) else "UNKNOWN",
    )
    add(
        "F-TF-001",
        "localization.map_to_odom",
        tf_status,
        "fresh dynamic map-to-odom transform",
        connected,
        [tf_ref, amcl_ref],
        checks=["inspect_transform_authorities", "inspect_localization_lifecycle"],
        hypotheses=["localization_tf_broadcast_disabled"]
        if broadcast is False and tf_status == "VIOLATED"
        else [],
    )
    if tf_status == "VIOLATED" and broadcast is False and _fresh(stamps.get(loc), now):
        propose(
            "R-TF-001",
            "localization.tf_broadcast",
            True,
            [tf_ref, amcl_ref],
            [
                "verify no competing authoritative map-to-odom publisher",
                "verify localization ACTIVE",
                "review minimal staging diff",
            ],
        )

    topic = requirements["interface_names"].get("sensing.lidar")
    signals = [s for s in model.signals if s.topic == topic]
    scan_ref = evidence("/signals", [s.model_dump(mode="json") for s in signals])
    scan_status = "UNKNOWN"
    age = None
    if len(signals) == 1 and model.completeness.get("signals") is True:
        sig = signals[0]
        if _fresh(sig.captured_at.isoformat(), now) and _number(sig.last_message_age_ms):
            age = sig.last_message_age_ms + (now - sig.captured_at).total_seconds() * 1000
            scan_status = "SATISFIED" if age <= min(500, sig.max_age_ms) else "VIOLATED"
    add(
        "F-SCAN-001",
        "obstacle_observation.fresh",
        scan_status,
        "fresh lidar within 500 ms cumulative age",
        age,
        [scan_ref],
        checks=["inspect_sensor_bridge_and_publishers"],
        hypotheses=["sensor_pipeline_paused_or_lost"] if scan_status == "VIOLATED" else [],
    )
    # A stale scan alone does not prove a paused bridge: no automatic bridge remedy.

    inactive = []
    lifecycle_known = model.completeness.get("lifecycle") is True
    for _role, node in sorted(requirements["lifecycle_nodes"].items()):
        rows = [r for r in model.lifecycle if r.name == node]
        if (
            len(rows) != 1
            or not _fresh(rows[0].captured_at.isoformat(), now)
            or rows[0].source != node + "/get_state"
            or rows[0].state == "UNKNOWN"
        ):
            lifecycle_known = False
        elif rows[0].state != "ACTIVE":
            inactive.append(node)
    life_ref = evidence("/lifecycle", [r.model_dump(mode="json") for r in model.lifecycle])
    add(
        "F-LIFE-001",
        "navigation.required_nodes_active",
        "VIOLATED"
        if inactive and lifecycle_known
        else "SATISFIED"
        if lifecycle_known
        else "UNKNOWN",
        "all selected navigation roles ACTIVE",
        inactive,
        [life_ref],
        checks=["inspect_managed_nodes_and_transition_failures"],
    )
    # Root causes are not inferred merely from inactive state.

    geometry = review["candidate"].get("geometry") or {}
    physical_radius = geometry.get("physical_radius_m")
    footprint_states = []
    footprint_refs = []
    undersized = []
    for costmap in requirements["costmap_topics"]:
        matches = [t for t in model.graph.get("topics", []) if t.get("name") == costmap]
        pubs = matches[0].get("publishers", []) if len(matches) == 1 else []
        node = pubs[0] if len(pubs) == 1 else None
        values = params.get(node, {})
        ref = evidence(
            _pointer("observations", "node_parameters", node),
            {k: values.get(k) for k in ["robot_radius", "footprint", "robot_base_frame"]},
            stamps.get(node),
        )
        footprint_refs.append(ref)
        try:
            polygon = json.loads(values.get("footprint", "null"))
        except (TypeError, ValueError):
            polygon = None
        radius = values.get("robot_radius")
        state = "UNKNOWN"
        if (
            node
            and _fresh(stamps.get(node), now)
            and geometry.get("complete") is True
            and _number(physical_radius)
            and physical_radius > 0
            and polygon == []
            and values.get("robot_base_frame") == frames.get("base")
            and _number(radius)
            and radius > 0
        ):
            state = "VIOLATED" if radius < physical_radius else "SATISFIED"
            if state == "VIOLATED":
                undersized.append(costmap)
        footprint_states.append(state)
    footprint_refs.append(evidence("/candidate/geometry", geometry, classification="INFERRED"))
    footprint_status = (
        "VIOLATED"
        if "VIOLATED" in footprint_states
        else "SATISFIED"
        if footprint_states and all(s == "SATISFIED" for s in footprint_states)
        else "UNKNOWN"
    )
    add(
        "F-GEOM-001",
        "costmap.covers_original_URDF_envelope",
        footprint_status,
        "observed effective costmap geometry covers original URDF",
        {"undersized_costmaps": undersized, "physical_radius_m": physical_radius},
        footprint_refs,
        checks=["independent_full_L1_effective_footprint_review"],
    )
    if undersized:
        propose(
            "R-GEOM-001",
            "costmap.radius",
            physical_radius,
            footprint_refs,
            [
                "review conservative padding and actual effective footprint",
                "apply only to observed undersized costmaps",
                "preserve original URDF and Body",
            ],
        )

    routes = []
    route_bad = False
    route_known = bool(model.completeness.get("graph"))
    for name, publisher, subscriber in [
        (
            requirements["velocity_input_topic"],
            requirements["lifecycle_nodes"]["controller"],
            requirements["lifecycle_nodes"]["collision_monitor"],
        ),
        (
            requirements["velocity_output_topic"],
            requirements["lifecycle_nodes"]["collision_monitor"],
            requirements["velocity_consumer_node"],
        ),
    ]:
        matches = [t for t in model.graph.get("topics", []) if t.get("name") == name]
        if len(matches) != 1:
            route_known = False
            continue
        t = matches[0]
        routes.append(t)
        if "publishers" not in t or "subscribers" not in t:
            route_known = False
        elif t["publishers"] != [publisher] or t["subscribers"] != [subscriber]:
            route_bad = True
    route_ref = evidence("/graph/topics", routes)
    add(
        "F-ROUTE-001",
        "velocity_chain.exclusive",
        "VIOLATED" if route_known and route_bad else "SATISFIED" if route_known else "UNKNOWN",
        "explicit unique selected publisher and subscriber per velocity edge",
        routes,
        [route_ref],
        checks=["identify_ownership_of_competing_endpoints", "review_authorized_control_topology"],
    )
    if route_bad:
        propose(
            "R-ROUTE-001",
            "control.route_review",
            None,
            [route_ref],
            [
                "identify competing endpoint ownership; do not disable by name",
                "operator-reviewed staging policy; preserve authorized route",
            ],
        )
    payload = {
        "snapshot_hash": model.snapshot_hash,
        "reviewed_at": now.isoformat(),
        "evidence_domain": evidence_domain,
        "run_id": run_id,
        "body_hash": body_hash,
        "source_urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
        "source_status": source_status,
        "constraints": constraints,
        "repair_candidates": repairs,
        "evidence_index": index,
        "derived_documents": derived_documents,
        "navigation_observation_status": review["status"],
        "diagnosis_issue_codes": sorted({i["issue_code"] for i in diagnosis["issues"]}),
    }
    case = RosEngineeringCaseV1(case_hash="", **payload)
    sealed = case.model_dump(mode="json")
    sealed.pop("case_hash")
    return case.model_copy(update={"case_hash": content_hash("roscase", sealed)})


def summarize_case(case: RosEngineeringCaseV1) -> dict:
    """Small decision card, with full original evidence retained in the IR."""
    return {
        "case_hash": case.case_hash,
        "mission": case.mission,
        "scope": case.evidence_domain,
        "authority": case.current_authority,
        "L1": case.current_l1,
        "source_status": case.source_status,
        "constraints": [
            {
                "id": c.id,
                "invariant": c.invariant,
                "expected": c.expected,
                "hypotheses": c.hypotheses,
                "status": c.status,
                "observed": c.observed if c.id not in {"F-ROUTE-001"} else "see evidence refs",
                "refs": c.evidence_refs,
                "next": c.next_checks,
            }
            for c in case.constraints
            if c.status != "SATISFIED"
        ],
        "candidate_ids": [r.id for r in case.repair_candidates],
        "conditional_candidates": [
            {
                "id": r.id,
                "semantic_field": r.semantic_field,
                "proposed_value": r.proposed_value,
                "classification": r.classification,
                "applicable_now": r.applicable_now,
                "stage_only": r.stage_only,
                "refs": r.evidence_refs,
                "preconditions": r.preconditions,
            }
            for r in case.repair_candidates
        ],
        "proof_obligations": [
            "new World and full fresh L1",
            "two canonical goals",
            "independent stop/contact/source review",
        ],
    }
